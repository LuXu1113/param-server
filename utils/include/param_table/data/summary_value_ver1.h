#ifndef UTILS_INCLUDE_PARAM_TABLE_DATA_SUMMARY_VALUE_VER1_H_
#define UTILS_INCLUDE_PARAM_TABLE_DATA_SUMMARY_VALUE_VER1_H_

#include "runtime/config_manager.h"

namespace ps {
namespace param_table {

struct SummaryValueVer1 {
  float n_;
  float sum_;
  float squared_sum_;
};

int summary_value_ver1_pull(SummaryValueVer1 *pull, const SummaryValueVer1& value);
int summary_value_ver1_push(SummaryValueVer1 *value, const SummaryValueVer1& grad, const ps::runtime::TrainingRule& rule);

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_DATA_SUMMARY_VALUE_VER1_H_

