#include "param_table/summary_value_ver1_table.h"

#include <stdio.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include <algorithm>
#include <butil/logging.h>
#include "absl/strings/str_format.h"
#include "message/types.h"
#include "toolkit/archive.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/rpc_agent.h"
#include "toolkit/thread_group.h"
#include "runtime/config_manager.h"

using std::vector;
using std::string;
using std::atomic;
using std::unique_ptr;
using std::shared_ptr;

using ps::ParamServerRequest;
using ps::ParamServerResponse;

using ps::toolkit::BinaryArchive;
using ps::toolkit::MPIAgent;
using ps::toolkit::RPCAgent;

using ps::runtime::ShardInfo;
using ps::runtime::ConfigManager;

namespace ps {
namespace param_table {

static BinaryArchive& operator<<(BinaryArchive& ar, const SummaryValueVer1& val) {
  ar << val.n_ << val.sum_ << val.squared_sum_;
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, SummaryValueVer1& val) {
  ar >> val.n_ >> val.sum_ >> val.squared_sum_;
  return ar;
}

static BinaryArchive& operator<<(BinaryArchive& ar, const vector<SummaryValueVer1>& p) {
  ar << (size_t)p.size();
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, vector<SummaryValueVer1>& p) {
  p.resize(ar.get<size_t>());
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

SummaryValueVer1Shard::SummaryValueVer1Shard() :
  data_(),
  begin_(0),
  end_(0),
  rw_mutex_() {
}

SummaryValueVer1Shard::~SummaryValueVer1Shard() {
}

uint64_t SummaryValueVer1Shard::mem_size() {
  uint64_t res = 0;
  rw_mutex_.ReaderLock();
  res = data_.capacity() * sizeof(SummaryValueVer1);
  rw_mutex_.ReaderUnlock();
  return res;
}

uint64_t SummaryValueVer1Shard::size() {
  uint64_t res = 0;
  rw_mutex_.ReaderLock();
  res = data_.size();
  rw_mutex_.ReaderUnlock();
  return res;
}

int SummaryValueVer1Shard::resize(uint64_t begin, uint64_t end) {
  int ret = ps::message::SUCCESS;
  rw_mutex_.WriterLock();
  begin_ = begin;
  end_ = end;
  data_.resize(end - begin);
  rw_mutex_.WriterUnlock();
  return ret;
}

int SummaryValueVer1Shard::assign(const vector<SummaryValueVer1>&value) {
  int ret = ps::message::SUCCESS;
  CHECK(end_ - begin_ == data_.size());
  rw_mutex_.WriterLock();
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] = value[(begin_ + i)];
  }
  rw_mutex_.WriterUnlock();
  return ret;
}

int SummaryValueVer1Shard::push(const vector<SummaryValueVer1>&value) {
  int ret = ps::message::SUCCESS;
  CHECK(end_ - begin_ == data_.size());
  rw_mutex_.WriterLock();
  for (size_t i = 0; i < data_.size(); ++i) {
    ret = summary_value_ver1_push(&(data_[i]), value[begin_ + i], ConfigManager::pick_training_rule());
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }
  rw_mutex_.WriterUnlock();
  return ret;
}

int SummaryValueVer1Shard::pull(vector<SummaryValueVer1> *value) {
  int ret = ps::message::SUCCESS;
  CHECK(end_ - begin_ == data_.size());
  // rw_mutex_.ReaderLock();
  for (size_t i = 0; i < data_.size(); ++i) {
    ret = summary_value_ver1_pull(&((*value)[begin_ + i]), data_[i]);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }
  // rw_mutex_.ReaderUnlock();
  return ret;
}

SummaryValueVer1Table::SummaryValueVer1Table() :
  name_(""),
  size_(0),
  shard_(32) {
}

SummaryValueVer1Table::SummaryValueVer1Table(const string& name) :
  name_(name),
  size_(0),
  shard_(32) {
}

SummaryValueVer1Table::~SummaryValueVer1Table() {
}

const string& SummaryValueVer1Table::name() const {
  return name_;
}

uint64_t SummaryValueVer1Table::size() const {
  return size_;
}

uint64_t SummaryValueVer1Table::mem_size() {
  uint64_t res = 0;
  for (size_t i = 0; i < shard_.size(); ++i) {
    res += shard_[i].mem_size();
  }
  return 0;
}

int SummaryValueVer1Table::resize(uint64_t size) {
  int ret = ps::message::SUCCESS;
  size_ = size;

  uint64_t total = 0;
  for (size_t i = 0; i < shard_.size(); ++i) {
    uint64_t length = size_ / shard_.size() + ((i % shard_.size() < (size_ % shard_.size())) ? 1 : 0);
    shard_[i].resize(total, total + length);
    total += length;
  }
  CHECK(total == size_);

  return ret;
}

int SummaryValueVer1Table::save(const string& path) {
  int ret = ps::message::SUCCESS;
  return ret;
}

int SummaryValueVer1Table::assign(const vector<SummaryValueVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(value.size() == size_);
  for (size_t i = 0; i < shard_.size(); ++i) {
    ret = shard_[i].assign(value);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }

  return ret;
}

int SummaryValueVer1Table::push(const vector<SummaryValueVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(value.size() == size_);
  for (size_t i = 0; i < shard_.size(); ++i) {
    ret = shard_[i].push(value);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }

  return ret;
}

int SummaryValueVer1Table::pull(vector<SummaryValueVer1> *value) {
  int ret = ps::message::SUCCESS;

  value->resize(size_);
  for (size_t i = 0; i < shard_.size(); ++i) {
    ret = shard_[i].pull(value);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }

  return ret;
}

uint64_t SummaryValueVer1TableServer::mem_size() {
  uint64_t res = 0;
  for (auto iter = tables_.begin(); iter != tables_.end(); ++iter) {
    res += iter->second->mem_size();
  }
  return res;
}

SummaryValueVer1TableServer::SummaryValueVer1TableServer() :
  tables_() {
}

SummaryValueVer1TableServer::~SummaryValueVer1TableServer() {
  for (auto iter = tables_.begin(); iter != tables_.end(); ++iter) {
    delete (iter->second);
  }
}

int SummaryValueVer1TableServer::create(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;

  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = ps::message::REGIST_EXISTING_SUMMARY_TABLE;
  } else {
    tables_[table_name] = new SummaryValueVer1Table(table_name);
    if (NULL == tables_[table_name]) {
      ret = ps::message::CAN_NOT_ALLOCATE_MEMORY;
    }
  }
  response->set_return_value(ret);
  LOG(INFO) << "create summary table: " << table_name << ", ret = " << ret;

  return ret;
}

int SummaryValueVer1TableServer::resize(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    uint64_t new_size;
    ar >> new_size;

    ret = iter->second->resize(new_size);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SUMMARY_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SummaryValueVer1TableServer::save(const ParamServerRequest& request, ParamServerResponse *response) const {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = iter->second->save(request.message());
  } else {
    ret = ps::message::PICK_NONEXISTENT_SUMMARY_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SummaryValueVer1TableServer::assign(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    vector<SummaryValueVer1> new_value;
    ar >> new_value;

    ret = iter->second->assign(new_value);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SUMMARY_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SummaryValueVer1TableServer::push(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    vector<SummaryValueVer1> push_value;
    ar >> push_value;

    ret = iter->second->push(push_value);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SUMMARY_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SummaryValueVer1TableServer::pull(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    vector<SummaryValueVer1> pull_value;

    ret = iter->second->pull(&pull_value);
    if (ret == ps::message::SUCCESS) {
      CHECK(iter->second->size() == pull_value.size());
      BinaryArchive oar;
      oar << pull_value;

      string message;
      oar.release(&message);

      response->set_message(message);
    }
  } else {
    ret = ps::message::PICK_NONEXISTENT_SUMMARY_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

SummaryValueVer1TableClient::SummaryValueVer1TableClient() :
  name_(""),
  size_(0),
  boundaries_() {
}

const string& SummaryValueVer1TableClient::name() const {
  return name_;
}

const uint64_t SummaryValueVer1TableClient::size() const {
  return size_;
}

int SummaryValueVer1TableClient::create(const string& name) {
  int ret = 0;
  name_ = name;

  DLOG(INFO) << "create summary table: " << name_;
  if (MPIAgent::mpi_rank_group() == 0) {
    ParamServerRequest request;
    vector<ParamServerResponse> response;
    request.set_message_type(ps::message::SUMMARY_TABLE_VER1_CREATE);
    request.set_table_name(name_);

    ret = RPCAgent::send_to_all(request, &response);
    if (0 != ret) {
      LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_CREATE, ret = " << ret;
    } else {
      for (size_t i = 0; i < response.size(); ++i) {
        ret = response[i].return_value();
        if (ps::message::SUCCESS != ret) {
          LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret)
                     << ", message_type = " << request.message_type()
                     << ", table_name = " << request.table_name();
          continue;
        }
      }
    }
  }

  return ret;
}

int SummaryValueVer1TableClient::resize(const uint64_t size) {
  int ret = 0;
  size_ = size;
  size_t mpi_size = MPIAgent::mpi_size_group();

  boundaries_.resize(mpi_size + 1);
  for (size_t i = 0; i <= mpi_size; ++i) {
    boundaries_[i] = i * size_ / mpi_size;
  }

  DLOG(INFO) << "resize summary table: " << name_ << ", size = " << size;
  if (MPIAgent::mpi_rank_group() == 0) {
    ParamServerRequest request;
    ParamServerResponse response;
    request.set_message_type(ps::message::SUMMARY_TABLE_VER1_RESIZE);
    request.set_table_name(name_);

    for (size_t i = 0; i < mpi_size; ++i) {
      uint64_t shard_size = boundaries_[i + 1] -  boundaries_[i];

      BinaryArchive ar;
      ar << shard_size;

      string message;
      ar.release(&message);

      request.set_message(message);
      ret = RPCAgent::send_to_one(request, &response, i);
      if (0 != ret) {
        LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_RESIZE, ret = " << ret;
        continue;
      }
      ret = response.return_value();
      if (ps::message::SUCCESS != ret) {
        LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret)
                   << ", message_type = " << request.message_type()
                   << ", table_name = " << request.table_name();
        continue;
      }
    }
  }

  return ret;
}

static void handle_async_save_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(FATAL) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
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
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SummaryValueVer1TableClient::save(const string& path) const {
  int ret = 0;

  LOG(INFO) << "save summary table: " << name_ << ", path = " << path;
  if (MPIAgent::mpi_rank_group() == 0) {
    size_t mpi_size = MPIAgent::mpi_size_group();
    atomic<int> count(mpi_size);

    ParamServerRequest request;
    request.set_message_type(ps::message::SUMMARY_TABLE_VER1_SAVE);
    request.set_table_name(name_);
    request.set_message(path);

    for (size_t i = 0; i < mpi_size; ++i) {
      ParamServerResponse *response = new ParamServerResponse();
      brpc::Controller *cntl = new brpc::Controller();
      google::protobuf::Closure *done = brpc::NewCallback(&handle_async_save_response, cntl, response, i, &count);

      ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
      if (0 != ret) {
        LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_SAVE, ret = " << ret;
        continue;
      }
    }

    while (count > 0) {
      usleep(5000);
    }
  }
  LOG(INFO) << "finish save summary table: " << name_ << ", path = " << path;

  return ret;
}

static void handle_async_assign_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id, atomic<int> *count) {
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
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SummaryValueVer1TableClient::assign(const vector<SummaryValueVer1>& value) const {
  int ret = 0;

  DLOG(INFO) << "assign summary table: " << name_;
  if (MPIAgent::mpi_rank_group() == 0) {
    CHECK(size_ == (uint64_t)value.size());

    size_t mpi_size = MPIAgent::mpi_size_group();
    atomic<int> count(mpi_size);

    for (size_t i = 0; i < mpi_size; ++i) {
      ParamServerRequest request;
      ParamServerResponse *response = new ParamServerResponse();
      request.set_message_type(ps::message::SUMMARY_TABLE_VER1_ASSIGN);
      request.set_table_name(name_);

      vector<SummaryValueVer1> shard;
      shard.assign(value.begin() + boundaries_[i], value.begin() + boundaries_[i + 1]);

      BinaryArchive ar;
      ar << shard;

      string message;
      ar.release(&message);
      request.set_message(message);

      brpc::Controller *cntl = new brpc::Controller();
      google::protobuf::Closure *done = brpc::NewCallback(&handle_async_assign_response, cntl, response, i, &count);

      ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
      if (0 != ret) {
        LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_ASSIGN, ret = " << ret;
        continue;
      }
    }

    while (count > 0) {
      usleep(5000);
    }
  }

  return ret;
}

static void handle_async_push_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id) {
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

int SummaryValueVer1TableClient::push(const vector<SummaryValueVer1>& value) const {
  int ret = 0;

  CHECK(size_ == (uint64_t)value.size());
  size_t mpi_size = MPIAgent::mpi_size_group();

  DLOG(INFO) << "async push summary table: " << name_;
  for (size_t i = 0; i < mpi_size; ++i) {
    ParamServerRequest request;
    ParamServerResponse *response = new ParamServerResponse();
    request.set_message_type(ps::message::SUMMARY_TABLE_VER1_PUSH);
    request.set_table_name(name_);

    vector<SummaryValueVer1> shard;
    shard.assign(value.begin() + boundaries_[i], value.begin() + boundaries_[i + 1]);

    BinaryArchive ar;
    ar << shard;

    string message;
    ar.release(&message);
    request.set_message(message);

    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_push_response, cntl, response, i);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
    if (0 != ret) {
      LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_PUSH, ret = " << ret;
      continue;
    }
  }
  return ret;
}

static void handle_async_pull_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id,
    vector<SummaryValueVer1>::iterator begin, vector<SummaryValueVer1>::iterator end, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
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
      vector<SummaryValueVer1> shard;
      BinaryArchive ar;
      ar.set_read_buffer(response->message());
      ar >> shard;

      CHECK(shard.size() == (size_t)(end - begin));
      for (size_t i = 0; i < shard.size(); ++i) {
        *(begin + i) = shard[i];
      }
    }
  }
  if (NULL != count) {
    --(*count);
  }

  return;
}
int SummaryValueVer1TableClient::pull(vector<SummaryValueVer1> *value) const {
  int ret = 0;

  value->resize(size_);
  size_t mpi_size = MPIAgent::mpi_size_group();
  atomic<int> count(mpi_size);

  DLOG(INFO) << "async pull summary table: " << name_;
  for (size_t i = 0; i < mpi_size; ++i) {
    ParamServerRequest request;
    ParamServerResponse *response = new ParamServerResponse();

    request.set_message_type(ps::message::SUMMARY_TABLE_VER1_PULL);
    request.set_table_name(name_);

    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_pull_response, cntl, response, i,
      value->begin() + boundaries_[i], value->begin() + boundaries_[i + 1], &count);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl,  done);
    if (0 != ret) {
      LOG(FATAL) << "rpc call SUMMARY_TABLE_VER1_PULL, ret = " << ret;
      continue;
    }
  }

  while (count > 0) {
    usleep(5000);
  }

  return ret;
}

} // namespace param_table
} // namespace ps

