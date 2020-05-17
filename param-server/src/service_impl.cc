#include "service_impl.h"

#include <butil/logging.h>
#include <brpc/server.h>
#include "absl/time/time.h"

#include "utils/proto/ps.pb.h"
#include "message/types.h"

namespace ps {
namespace param_server {

void ParamServerServiceImpl::remote_call(google::protobuf::RpcController *cntl_base,
                                         const ParamServerRequest *request,
                                         ParamServerResponse *response,
                                         google::protobuf::Closure *done) {
  // This object helps you to call done->Run() in RAII style. If you need
  // to process the request asynchronously, pass done_guard.release().
  brpc::ClosureGuard done_guard(done);

  int ret = ps::message::SUCCESS;
  uint32_t message_type = request->message_type();
  absl::Time ts1;
  absl::Time ts2;
  switch (message_type) {
   case ps::message::SPARSE_TABLE_VER1_CREATE:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.create(*request, response);
    ts2 = absl::Now();
    sparse_table_create_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_SAVE:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.save(*request, response);
    ts2 = absl::Now();
    sparse_table_save_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_ASSIGN:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.assign(*request, response);
    ts2 = absl::Now();
    sparse_table_assign_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_PULL:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.pull(*request, response);
    ts2 = absl::Now();
    sparse_table_pull_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_PUSH:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.push(*request, response);
    ts2 = absl::Now();
    sparse_table_push_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_TIME_DECAY:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.time_decay(*request, response);
    ts2 = absl::Now();
    sparse_table_time_decay_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_SHRINK:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.shrink(*request, response);
    ts2 = absl::Now();
    sparse_table_shrink_log_.record(ts1, ts2);
    break;

   case ps::message::SPARSE_TABLE_VER1_FEATURE_NUM:
    ts1 = absl::Now();
    ret = sparse_kv_ver1_table_server_.feature_num(*request, response);
    ts2 = absl::Now();
    sparse_table_feature_num_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_CREATE:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.create(*request, response);
    ts2 = absl::Now();
    embedding_table_create_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_SAVE:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.save(*request, response);
    ts2 = absl::Now();
    embedding_table_save_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_ASSIGN:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.assign(*request, response);
    ts2 = absl::Now();
    embedding_table_assign_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_PULL:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.pull(*request, response);
    ts2 = absl::Now();
    embedding_table_pull_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_PUSH:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.push(*request, response);
    ts2 = absl::Now();
    embedding_table_push_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_TIME_DECAY:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.time_decay(*request, response);
    ts2 = absl::Now();
    embedding_table_time_decay_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_SHRINK:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.shrink(*request, response);
    ts2 = absl::Now();
    embedding_table_shrink_log_.record(ts1, ts2);
    break;

   case ps::message::EMBEDDING_TABLE_VER1_FEATURE_NUM:
    ts1 = absl::Now();
    ret = embedding_ver1_table_server_.feature_num(*request, response);
    ts2 = absl::Now();
    embedding_table_feature_num_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_CREATE:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.create(*request, response);
    ts2 = absl::Now();
    dense_table_create_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_SAVE:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.save(*request, response);
    dense_table_save_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_ASSIGN:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.assign(*request, response);
    ts2 = absl::Now();
    dense_table_assign_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_PULL:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.pull(*request, response);
    ts2 = absl::Now();
    dense_table_pull_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_PUSH:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.push(*request, response);
    ts2 = absl::Now();
    dense_table_push_log_.record(ts1, ts2);
    break;

   case ps::message::DENSE_TABLE_VER1_RESIZE:
    ts1 = absl::Now();
    ret = dense_value_ver1_table_server_.resize(*request, response);
    ts2 = absl::Now();
    dense_table_resize_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_CREATE:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.create(*request, response);
    ts2 = absl::Now();
    summary_table_create_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_SAVE:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.save(*request, response);
    ts2 = absl::Now();
    summary_table_save_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_ASSIGN:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.assign(*request, response);
    ts2 = absl::Now();
    summary_table_assign_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_PULL:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.pull(*request, response);
    ts2 = absl::Now();
    summary_table_pull_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_PUSH:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.push(*request, response);
    ts2 = absl::Now();
    summary_table_push_log_.record(ts1, ts2);
    break;

   case ps::message::SUMMARY_TABLE_VER1_RESIZE:
    ts1 = absl::Now();
    ret = summary_value_ver1_table_server_.resize(*request, response);
    ts2 = absl::Now();
    summary_table_resize_log_.record(ts1, ts2);
    break;

   case ps::message::SHUTDOWN:
    ret = shutdown();
    response->set_return_value(ret);
    break;

   default:
    LOG(FATAL) << "message type invalid: " << message_type;
    response->set_return_value(ps::message::MESSAGE_TYPE_INVALID);
    break;
  }
}

int ParamServerServiceImpl::shutdown() {
  int ret = ps::message::SUCCESS;
  has_shutdown_ = true;

  sparse_table_create_log_.set_name("sparse_table_create");
  sparse_table_save_log_.set_name("sparse_table_save");
  sparse_table_assign_log_.set_name("sparse_table_assign");
  sparse_table_pull_log_.set_name("sparse_table_pull");
  sparse_table_push_log_.set_name("sparse_table_push");
  sparse_table_time_decay_log_.set_name("sparse_table_time_decay");
  sparse_table_shrink_log_.set_name("sparse_table_shrink");
  sparse_table_feature_num_log_.set_name("sparse_table_feature_num");
  embedding_table_create_log_.set_name("embedding_table_create");
  embedding_table_save_log_.set_name("embedding_table_save");
  embedding_table_assign_log_.set_name("embedding_table_assing");
  embedding_table_pull_log_.set_name("embedding_table_pull");
  embedding_table_push_log_.set_name("embedding_table_push");
  embedding_table_time_decay_log_.set_name("embedding_table_time_decay");
  embedding_table_shrink_log_.set_name("embedding_table_shrink");
  embedding_table_feature_num_log_.set_name("embedding_table_feature_num");
  dense_table_create_log_.set_name("dense_table_create");
  dense_table_save_log_.set_name("dense_table_save");
  dense_table_assign_log_.set_name("dense_table_assign");
  dense_table_pull_log_.set_name("dense_table_pull");
  dense_table_push_log_.set_name("dense_table_push");
  dense_table_resize_log_.set_name("dense_table_resize");
  summary_table_create_log_.set_name("summary_table_create");
  summary_table_save_log_.set_name("summary_table_save");
  summary_table_assign_log_.set_name("summary_table_assign");
  summary_table_pull_log_.set_name("summary_table_pull");
  summary_table_push_log_.set_name("summary_table_push");
  summary_table_resize_log_.set_name("summary_table_resize");

  sparse_table_create_log_.log();
  sparse_table_save_log_.log();
  sparse_table_assign_log_.log();
  sparse_table_pull_log_.log();
  sparse_table_push_log_.log();
  sparse_table_time_decay_log_.log();
  sparse_table_shrink_log_.log();
  sparse_table_feature_num_log_.log();
  embedding_table_create_log_.log();
  embedding_table_save_log_.log();
  embedding_table_assign_log_.log();
  embedding_table_pull_log_.log();
  embedding_table_push_log_.log();
  embedding_table_time_decay_log_.log();
  embedding_table_shrink_log_.log();
  embedding_table_feature_num_log_.log();
  dense_table_create_log_.log();
  dense_table_save_log_.log();
  dense_table_assign_log_.log();
  dense_table_pull_log_.log();
  dense_table_push_log_.log();
  dense_table_resize_log_.log();
  summary_table_create_log_.log();
  summary_table_save_log_.log();
  summary_table_assign_log_.log();
  summary_table_pull_log_.log();
  summary_table_push_log_.log();
  summary_table_resize_log_.log();

  return ret;
}

bool ParamServerServiceImpl::has_shutdown() {
  return has_shutdown_;
}

} // namespace param_server
} // namespace ps

