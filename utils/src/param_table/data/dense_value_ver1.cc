#include "param_table/data/dense_value_ver1.h"

#include "message/types.h"

using ps::runtime::TrainingRule;

namespace ps {
namespace param_table {

int dense_value_ver1_pull(DenseValueVer1Pull *pull, const DenseValueVer1& value) {
  int ret = ps::message::SUCCESS;
  (*pull).weight_ = value.weight_;
  return ret;
}

int dense_value_ver1_push(DenseValueVer1 *value, const DenseValueVer1Push& grad, const TrainingRule& rule) {
  int ret = ps::message::SUCCESS;

  if (rule.dense_.optimizer_ == "" || rule.dense_.optimizer_ == "base") {
    float wd_ = rule.dense_.weight_decay_ * value->weight_;

    value->step_ += 1.0;
    value->momentum_ = rule.dense_.mom_decay_rate_ * value->momentum_ + (1.0 - rule.dense_.mom_decay_rate_) * grad.weight_;
    value->ada_d2sum_ = rule.dense_.ada_decay_rate_ * value->ada_d2sum_ + 1.0;
    value->ada_g2sum_ = rule.dense_.ada_decay_rate_ * value->ada_g2sum_ + grad.weight_ * grad.weight_;

    float m_ = value->momentum_;
    float v_ = value->ada_g2sum_ / value->ada_d2sum_;
    value->weight_ += rule.dense_.learning_rate_ * sqrt((1.0 + rule.dense_.ada_epsilon_) / (v_ + rule.dense_.ada_epsilon_)) * m_ - wd_;

  } else if (rule.dense_.optimizer_ == "AdamW") {
    float wd_ = rule.dense_.weight_decay_ * value->weight_;

    value->step_ += 1.0;
    value->momentum_ = rule.dense_.mom_decay_rate_ * value->momentum_ + (1.0 - rule.dense_.mom_decay_rate_) * grad.weight_;
    value->ada_g2sum_ = rule.dense_.ada_decay_rate_ * value->ada_g2sum_ + (1.0 - rule.dense_.ada_decay_rate_) * grad.weight_ * grad.weight_;
    value->power_ada_beta_1_ *= rule.dense_.mom_decay_rate_;
    value->power_ada_beta_2_ *= rule.dense_.ada_decay_rate_;

    float m_ = value->momentum_ / (1.0 - value->power_ada_beta_1_);
    float v_ = value->ada_g2sum_ / (1.0 - value->power_ada_beta_2_);
    value->weight_ += rule.dense_.learning_rate_ / (sqrt(v_) + rule.dense_.ada_epsilon_) * m_ - wd_;

  } else if (rule.dense_.optimizer_ == "RMSProp") {
    float wd_ = rule.dense_.weight_decay_ * value->weight_;

    value->step_ += 1.0;
    value->momentum_ = rule.dense_.mom_decay_rate_ * value->momentum_ + (1.0 - rule.dense_.mom_decay_rate_) * grad.weight_;
    value->ada_g2sum_ = rule.dense_.ada_decay_rate_ * value->ada_g2sum_ + (1.0 - rule.dense_.ada_decay_rate_) * grad.weight_ * grad.weight_;

    float m_ = value->momentum_ / (1.0 - value->power_ada_beta_1_);
    float v_ = value->ada_g2sum_ / (1.0 - value->power_ada_beta_2_);
    value->weight_ += rule.dense_.learning_rate_ / (sqrt(v_) + rule.dense_.ada_epsilon_) * m_ - wd_;

  } else {
    ret = ps::message::UNKNOWN_OPTIMIZER;
    LOG(FATAL) << "unknown optimizer: " << rule.dense_.optimizer_;
  }

  return ret;
}

} // namespace param_table
} // namespace ps

