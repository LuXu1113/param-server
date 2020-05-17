#include "toolkit/rpc_agent.h"
#include <atomic>
#include <memory>
#include "message/types.h"

using std::vector;
using std::unique_ptr;
using brpc::Channel;
using brpc::Controller;
using google::protobuf::Closure;
using ps::ParamServerService_Stub;

namespace ps {
namespace toolkit {

static bool is_inited_ = false;
static vector<RPCServerInfo> servers_;
static vector<Channel *> channps_;
static vector<ParamServerService_Stub *> stubs_;
static brpc::CompressType compress_type_ = brpc::COMPRESS_TYPE_NONE;

int RPCAgent::initialize(const vector<RPCServerInfo>& servers) {
  if (is_inited_) {
    return -1;
  }

  servers_ = servers;

  brpc::ChannelOptions options;
  options.protocol           = "baidu_std";
  options.connection_type    = "single";
  options.connect_timeout_ms = 0x7fffffff;
  options.timeout_ms         = 500000;
  options.max_retry          = 3;

  channps_.resize(servers_.size());
  stubs_.resize(servers_.size());
  for (size_t i = 0; i < servers.size(); ++i) {
    channps_[i] = new Channel();
    if (channps_[i]->Init(servers[i].ip_.c_str(), servers[i].port_, &options) != 0) {
      LOG(ERROR) << "Fail to initialize channel, [ip:port] = ["
                 << servers[i].ip_ << ":" << servers[i].port_ << "]";
      for (size_t j = 0; j < 0; ++j) {
        delete(stubs_[j]);
        delete(channps_[j]);
      }
      stubs_.clear();
      channps_.clear();
      servers_.clear();
      return -1;
    } else {
      stubs_[i] = new ParamServerService_Stub(channps_[i]);
    }
  }
  compress_type_ = brpc::COMPRESS_TYPE_SNAPPY;
  is_inited_ = 1;

  return 0;
}

static void handle_async_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(ERROR) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
        << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
        << ", latency = " << cntl->latency_us() << "us";
    }
  }

  return;
}

int RPCAgent::shutdown() {
  int ret = 0;

  ParamServerRequest request;
  request.set_message_type(ps::message::SHUTDOWN);
  for (size_t i = 0; i < servers_.size(); ++i) {
    ParamServerResponse *response = new ParamServerResponse();
    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_response, cntl, response, i);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
    if (0 != ret) {
      LOG(ERROR) << "rpc call SHUTDOWN, ret = " << ret;
    }
  }

  return ret;
}

int RPCAgent::finalize() {
  if (!is_inited_) {
    return -1;
  }

  for (size_t i = 0; i < servers_.size(); ++i) {
    delete(stubs_[i]);
    stubs_[i] = NULL;
    delete(channps_[i]);
    channps_[i] = NULL;
  }
  stubs_.clear();
  channps_.clear();
  servers_.clear();

  is_inited_ = 0;

  return 0;
}

int RPCAgent::send_to_all(const ParamServerRequest& request, vector<ParamServerResponse> *response) {
  response->resize(servers_.size());
  for (size_t i = 0; i < servers_.size(); ++i) {
    Controller cntl;
    cntl.set_request_compress_type(compress_type_);
    cntl.set_response_compress_type(compress_type_);
    // cntl.request_attachment().append(attachment);
    stubs_[i]->remote_call(&cntl, &request, &((*response)[i]), NULL);
    if (cntl.Failed()) {
      LOG(ERROR) << "remote_call to " << servers_[i].ip_ << ":" << servers_[i].port_ << " fail, error text is:" << cntl.ErrorText();
      return -1;
    } else {
      DLOG(INFO) << "Received response from " << cntl.remote_side()
                 << ": " << (*response)[i].message() << " (attached = " << cntl.response_attachment() << ")"
                 << ", latency = " << cntl.latency_us() << "us, message type = " << request.message_type();
    }
  }
  return 0;
}

int RPCAgent::send_to_one(const ParamServerRequest& request, ParamServerResponse *response, size_t server_id) {
  CHECK(server_id < servers_.size());
  CHECK(response != NULL);

  Controller cntl;
  cntl.set_request_compress_type(compress_type_);
  cntl.set_response_compress_type(compress_type_);
  // cntl.request_attachment().append(attachment);
  stubs_[server_id]->remote_call(&cntl, &request, response, NULL);
  if (cntl.Failed()) {
    LOG(ERROR) << "remote_call to " << servers_[server_id].ip_ << ":" << servers_[server_id].port_ << " fail, error text is:" << cntl.ErrorText();
    return -1;
  } else {
    DLOG(INFO) << "Received response from " << cntl.remote_side()
               << ": " << response->message() << " (attached = " << cntl.response_attachment() << ")"
               << ", latency = " << cntl.latency_us() << "us, message type = " << request.message_type();
  }

  return 0;
}

int RPCAgent::send_to_one_async(const ParamServerRequest& request, ParamServerResponse *response, size_t server_id, Controller *cntl, google::protobuf::Closure *done) {
  CHECK(server_id < servers_.size());
  CHECK(response != NULL);

  cntl->set_request_compress_type(compress_type_);
  cntl->set_response_compress_type(compress_type_);
  // cntl->request_attachment().append(attachment);
  stubs_[server_id]->remote_call(cntl, &request, response, done);

  return 0;
}

} // namespace toolkit
} // namespace ps

