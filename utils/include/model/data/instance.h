#ifndef UTILS_INCLUDE_MODEL_DATA_INSTANCE_H_
#define UTILS_INCLUDE_MODEL_DATA_INSTANCE_H_

#include <vector>
#include <map>
#include <string>
#include "param_table/data/sparse_kv_ver1.h"
#include "param_table/data/sparse_embedding_ver1.h"

namespace ps {
namespace model {

struct Instance {
  std::string lineid_;
  std::vector<float> labps_; // multi-label task
  float adq_;
  float label_;
  float bid_;
  float prior_;

  ps::param_table::SparseKeyVer1 position_fea_;
  int position_idx_;

  int fea_num_;
  std::vector<ps::param_table::SparseFeatureVer1> feas_;
  std::vector<ps::param_table::SparseValueVer1>   fea_pulls_;
  std::vector<ps::param_table::SparseValueVer1>   fea_pushs_;

  int memory_fea_num_;
  std::vector<ps::param_table::SparseFeatureVer1> memory_feas_;
  std::vector<ps::param_table::SparseEmbeddingVer1> memory_fea_pulls_;
  std::vector<ps::param_table::SparseEmbeddingVer1> memory_fea_pushs_;

  std::map<std::string, std::vector<float> > vec_values_;

  float lr_output_, lr_pred_;
  float fm_output_, fm_pred_, fm_grad_;
  std::vector<float> fm_v_sums_;
  float mf_output_, mf_pred_;
  std::vector<float> mf_v_sums_;
  float wide_output_, wide_grad_;
  std::vector<float> dnn_preds_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DATA_INSTANCE_H_

