#ifndef UTILS_INCLUDE_PARAM_TABLE_DATA_DENSE_VALUE_VER1_H_
#define UTILS_INCLUDE_PARAM_TABLE_DATA_DENSE_VALUE_VER1_H_

#include "runtime/config_manager.h"

namespace ps {
namespace param_table {

struct DenseValueVer1 {
  float weight_;
  float momentum_;
  float ada_d2sum_;
  float ada_g2sum_;
  float power_ada_beta_1_;
  float power_ada_beta_2_;
  float max_g2sum_;
  float norm_grad_;
  float norm_weight_;
  int64_t step_;
};

struct DenseValueVer1Pull {
  float weight_;
};

struct DenseValueVer1Push {
  float weight_;
  float norm_grad_;
  float norm_weight_;
};

int dense_value_ver1_pull(DenseValueVer1Pull *pull, const DenseValueVer1& value);
int dense_value_ver1_push(DenseValueVer1 *value, const DenseValueVer1Push& grad, const ps::runtime::TrainingRule& rule);

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_DATA_DENSE_VALUE_VER1_H_

