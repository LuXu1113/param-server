#ifndef PARAM_SERVER_INCLUDE_SERVICE_IMPL_H_
#define PARAM_SERVER_INCLUDE_SERVICE_IMPL_H_

#include "utils/proto/ps.pb.h"
#include "toolkit/operating_log.h"
#include "param_table/dense_value_ver1_table.h"
#include "param_table/sparse_embedding_ver1_table.h"
#include "param_table/sparse_kv_ver1_table.h"
#include "param_table/summary_value_ver1_table.h"

namespace ps {
namespace param_server {

class ParamServerServiceImpl : public ParamServerService {
 public:
  ParamServerServiceImpl() :
    has_shutdown_(false) {
  }
  ParamServerServiceImpl(const ParamServerServiceImpl&) = delete;
  virtual ~ParamServerServiceImpl() {};
  virtual void remote_call(google::protobuf::RpcController *cntl_base,
                           const ParamServerRequest *request,
                           ParamServerResponse *response,
                           google::protobuf::Closure *done);
  bool has_shutdown();

 private:
  int shutdown();

  bool has_shutdown_;

  ps::param_table::SparseKVVer1TableServer        sparse_kv_ver1_table_server_;
  ps::param_table::SparseEmbeddingVer1TableServer embedding_ver1_table_server_;
  ps::param_table::DenseValueVer1TableServer      dense_value_ver1_table_server_;
  ps::param_table::SummaryValueVer1TableServer    summary_value_ver1_table_server_;

  ps::toolkit::OperatingLog sparse_table_create_log_;
  ps::toolkit::OperatingLog sparse_table_save_log_;
  ps::toolkit::OperatingLog sparse_table_assign_log_;
  ps::toolkit::OperatingLog sparse_table_pull_log_;
  ps::toolkit::OperatingLog sparse_table_push_log_;
  ps::toolkit::OperatingLog sparse_table_time_decay_log_;
  ps::toolkit::OperatingLog sparse_table_shrink_log_;
  ps::toolkit::OperatingLog sparse_table_feature_num_log_;
  ps::toolkit::OperatingLog embedding_table_create_log_;
  ps::toolkit::OperatingLog embedding_table_save_log_;
  ps::toolkit::OperatingLog embedding_table_assign_log_;
  ps::toolkit::OperatingLog embedding_table_pull_log_;
  ps::toolkit::OperatingLog embedding_table_push_log_;
  ps::toolkit::OperatingLog embedding_table_time_decay_log_;
  ps::toolkit::OperatingLog embedding_table_shrink_log_;
  ps::toolkit::OperatingLog embedding_table_feature_num_log_;
  ps::toolkit::OperatingLog dense_table_create_log_;
  ps::toolkit::OperatingLog dense_table_save_log_;
  ps::toolkit::OperatingLog dense_table_assign_log_;
  ps::toolkit::OperatingLog dense_table_pull_log_;
  ps::toolkit::OperatingLog dense_table_push_log_;
  ps::toolkit::OperatingLog dense_table_resize_log_;
  ps::toolkit::OperatingLog summary_table_create_log_;
  ps::toolkit::OperatingLog summary_table_save_log_;
  ps::toolkit::OperatingLog summary_table_assign_log_;
  ps::toolkit::OperatingLog summary_table_pull_log_;
  ps::toolkit::OperatingLog summary_table_push_log_;
  ps::toolkit::OperatingLog summary_table_resize_log_;
};

} // namespace param_server
} // namespace ps

