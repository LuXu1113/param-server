#include "param_table/data/summary_value_ver1.h"

#include "message/types.h"

using ps::runtime::TrainingRule;

namespace ps {
namespace param_table {

int summary_value_ver1_pull(SummaryValueVer1 *pull, const SummaryValueVer1& value) {
  int ret = ps::message::SUCCESS;
  (*pull) = value;
  return ret;
}

int summary_value_ver1_push(SummaryValueVer1 *value, const SummaryValueVer1& grad, const TrainingRule& rule) {
  int ret = ps::message::SUCCESS;

  if (grad.n_ > 0) {
    value->n_ = rule.dense_.summary_decay_rate_ * value->n_ + grad.n_;
    value->sum_ = rule.dense_.summary_decay_rate_ * value->sum_ + grad.sum_;
    value->squared_sum_ = rule.dense_.summary_decay_rate_ * value->squared_sum_ + grad.squared_sum_
                        + grad.n_ * rule.dense_.summary_squared_sum_epsilon_;
  }
  return ret;
}

} // namespace param_table
} // namespace ps

