#ifndef UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_KV_VER1_H_
#define UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_KV_VER1_H_

#include <vector>
#include <string>
#include "runtime/config_manager.h"

namespace ps {
namespace param_table {

typedef uint64_t SparseKeyVer1;
typedef uint32_t SparseSlotVer1;

struct SparseValueVer1 {
  SparseSlotVer1 slot_;
  int silent_days_;

  float show_;
  float clk_;

  float lr_w_;
  float lr_g2sum_;

  float fm_w_;
  float fm_w_g2sum_;
  std::vector<float> fm_v_;
  float fm_v_g2sum_;

  float mf_w_;
  float mf_w_g2sum_;
  std::vector<float> mf_v_;
  float mf_v_g2sum_;

  float wide_w_;
  float wide_g2sum_;

  uint64_t version_;
  float delta_score_;
};

struct SparseFeatureVer1 {
  SparseKeyVer1  sign_;
  SparseSlotVer1 slot_;
};

SparseValueVer1 sparse_value_ver1_default();
int sparse_value_ver1_init(SparseValueVer1 *value, const ps::runtime::TrainingRule& rule);
int sparse_value_ver1_push(SparseValueVer1 *value, const SparseValueVer1& grad, const ps::runtime::TrainingRule& rule);
int sparse_value_ver1_merge(SparseValueVer1 *value, const SparseValueVer1& new_value, const ps::runtime::TrainingRule& rule);
int sparse_value_ver1_to_string(const SparseKeyVer1& key, const SparseValueVer1& value, std::string *str);
int sparse_value_ver1_time_decay(SparseValueVer1 *value, const ps::runtime::TrainingRule& rule);
bool sparse_value_ver1_shrink(const SparseValueVer1& value, const ps::runtime::TrainingRule& rule);

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_KV_VER1_H_

