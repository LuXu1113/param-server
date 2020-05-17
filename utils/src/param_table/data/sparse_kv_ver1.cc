#include "param_table/data/sparse_kv_ver1.h"

#include <math.h>
#include <butil/logging.h>
#include "absl/strings/str_format.h"
#include "absl/random/random.h"

using std::string;
using ps::runtime::SparseTrainingRule;
using ps::runtime::TrainingRule;

namespace ps {
namespace param_table {

static inline void bound(float *x, const float lower_bound, const float upper_bound) {
  CHECK(!(lower_bound > upper_bound));
  if (*x < lower_bound) { *x = lower_bound;  }
  if (*x > upper_bound) { *x = upper_bound; }
}

static inline float random(const float initial_range) {
  static absl::BitGen gen;
  return absl::uniform_real_distribution<float>(-initial_range, initial_range)(gen);
}

SparseValueVer1 sparse_value_ver1_default() {
  SparseValueVer1 value;

  value.slot_ = -1;
  value.silent_days_ = 0;

  value.show_ = 0;
  value.clk_ = 0;

  value.lr_w_ = 0;
  value.lr_g2sum_ = 0;

  value.fm_w_ = 0;
  value.fm_w_g2sum_ = 0;
  value.fm_v_.clear();
  value.fm_v_g2sum_ = 0;

  value.mf_w_ = 0;
  value.mf_w_g2sum_ = 0;
  value.mf_v_.clear();
  value.mf_v_g2sum_ = 0;

  value.wide_w_ = 0;
  value.wide_g2sum_ = 0;
  value.version_ = 0;
  value.delta_score_ = 0;

  return value;
}

int sparse_value_ver1_init(SparseValueVer1 *value, const TrainingRule& rule) {
  int ret = 0;
  const SparseTrainingRule& conf = rule.sparse_;

  value->slot_ = -1;
  value->silent_days_ = 0;

  value->show_ = 0;
  value->clk_ = 0;

  value->lr_w_ = random(conf.lr_rule_.initial_range_);
  bound(&(value->lr_w_), conf.lr_rule_.weight_lower_bound_, conf.lr_rule_.weight_upper_bound_);
  value->lr_g2sum_ = 0;

  value->fm_w_ = random(conf.fm_rule_.initial_range_);
  bound(&(value->fm_w_), conf.fm_rule_.weight_lower_bound_, conf.fm_rule_.weight_upper_bound_);
  value->fm_w_g2sum_ = 0;
  value->fm_v_.resize(conf.fm_rule_.dim_);
  for (size_t i = 0; i < value->fm_v_.size(); ++i) {
    value->fm_v_[i] = random(conf.fm_rule_.initial_range_);
    bound(&(value->fm_v_[i]), conf.fm_rule_.weight_lower_bound_, conf.fm_rule_.weight_upper_bound_);
  }
  value->fm_v_g2sum_ = 0;

  value->mf_w_ = random(conf.mf_rule_.initial_range_);
  bound(&(value->mf_w_), conf.mf_rule_.weight_lower_bound_, conf.mf_rule_.weight_upper_bound_);
  value->mf_w_g2sum_ = 0;
  value->mf_v_.resize(conf.mf_rule_.dim_);
  for (size_t i = 0; i < value->mf_v_.size(); ++i) {
    value->mf_v_[i] = random(conf.mf_rule_.initial_range_);
    bound(&(value->mf_v_[i]), conf.mf_rule_.weight_lower_bound_, conf.mf_rule_.weight_upper_bound_);
  }
  value->mf_v_g2sum_ = 0;

  value->wide_w_ = random(conf.wide_rule_.initial_range_);
  bound(&(value->wide_w_), conf.wide_rule_.weight_lower_bound_, conf.wide_rule_.weight_upper_bound_);
  value->wide_g2sum_ = 0;

  value->version_ = 0;
  value->delta_score_ = 0;

  return ret;
}

static inline void adagrad(int n, float *w, const float *g, float g_scale, const float learning_rate,
                           float *g2sum, const float initial_g2sum,
                           const float weight_lower_bound, const float weight_upper_bound,
                           const bool version_aware, const uint64_t version_diff) {
  if (g_scale <= 0) {
    g_scale = 1.0;
  }
  if (version_aware && version_diff > 0) {
    g_scale *= sqrt(1.0 + version_diff);
  }
  float add_g2sum = 0.0;
  for (int i = 0; i < n; ++i) {
    float scaled_grad = g[i] / g_scale;
    w[i] += learning_rate * scaled_grad * sqrt(initial_g2sum / (initial_g2sum + (*g2sum)));
    bound(&(w[i]), weight_lower_bound, weight_upper_bound);
    add_g2sum += scaled_grad * scaled_grad;
  }

  (*g2sum) += add_g2sum / n;
}

int sparse_value_ver1_push(SparseValueVer1 *value, const SparseValueVer1& grad, const TrainingRule& rule) {
  int ret = 0;
  const SparseTrainingRule& conf = rule.sparse_;

  // CHECK(value->slot_ == grad.slot_) << "slot: " << value->slot_ << ", newslot: " << grad.slot_;
  value->silent_days_ = 0;

  value->show_ += grad.show_;
  value->clk_  += grad.clk_;

  // update
  uint64_t version_diff = value->version_ - grad.version_;

  // lr update
  adagrad(1, &(value->lr_w_), &(grad.lr_w_), grad.show_, conf.lr_rule_.learning_rate_,
          &(value->lr_g2sum_), conf.lr_rule_.initial_g2sum_,
          conf.lr_rule_.weight_lower_bound_, conf.lr_rule_.weight_upper_bound_,
          conf.lr_rule_.version_aware_, version_diff);

  // fm update
  adagrad(1, &(value->fm_w_), &(grad.fm_w_), grad.show_, conf.fm_rule_.learning_rate_,
          &(value->fm_w_g2sum_), conf.fm_rule_.initial_g2sum_,
          conf.fm_rule_.weight_lower_bound_, conf.fm_rule_.weight_upper_bound_,
          conf.fm_rule_.version_aware_, version_diff);
  if (!(grad.fm_v_.empty() || value->fm_v_.empty())) {
    adagrad(conf.fm_rule_.dim_, &(value->fm_v_[0]), &(grad.fm_v_[0]), grad.show_, conf.fm_rule_.learning_rate_,
            &(value->fm_v_g2sum_), conf.fm_rule_.initial_g2sum_,
            conf.fm_rule_.weight_lower_bound_, conf.fm_rule_.weight_upper_bound_,
            conf.fm_rule_.version_aware_, version_diff);
  }

  // mf update
  adagrad(1, &(value->mf_w_), &(grad.mf_w_), grad.show_, conf.mf_rule_.learning_rate_,
          &(value->mf_w_g2sum_), conf.mf_rule_.initial_g2sum_,
          conf.mf_rule_.weight_lower_bound_, conf.mf_rule_.weight_upper_bound_,
          conf.mf_rule_.version_aware_, version_diff);
  if (!(grad.mf_v_.empty() || value->mf_v_.empty())) {
    adagrad(conf.mf_rule_.dim_, &(value->mf_v_[0]), &(grad.mf_v_[0]), grad.show_, conf.mf_rule_.learning_rate_,
            &(value->mf_v_g2sum_), conf.mf_rule_.initial_g2sum_,
            conf.mf_rule_.weight_lower_bound_, conf.mf_rule_.weight_upper_bound_,
            conf.mf_rule_.version_aware_, version_diff);
  }

  // wide update
  adagrad(1, &(value->wide_w_), &(grad.wide_w_), grad.show_, conf.wide_rule_.learning_rate_,
          &(value->wide_g2sum_), conf.wide_rule_.initial_g2sum_,
          conf.wide_rule_.weight_lower_bound_, conf.wide_rule_.weight_upper_bound_,
          conf.wide_rule_.version_aware_, version_diff);

  ++(value->version_);
  value->delta_score_ += (grad.show_ - grad.clk_) * conf.nonclk_coeff_ + grad.clk_ * conf.clk_coeff_;

  return ret;
}

int sparse_value_ver1_merge(SparseValueVer1 *value, const SparseValueVer1& new_value, const ps::runtime::TrainingRule& rule) {
  int ret = 0;
  // const SparseTrainingRule& conf = rule.sparse_;

  value->show_ += new_value.show_;
  value->clk_  += new_value.clk_;

  value->lr_w_ += new_value.lr_w_;

  value->fm_w_ += new_value.fm_w_;
  for (size_t i = 0; i < value->fm_v_.size() && i < new_value.fm_v_.size(); ++i) {
    value->fm_v_[i] += new_value.fm_v_[i];
  }

  value->mf_w_ += new_value.mf_w_;
  for (size_t i = 0; i < value->mf_v_.size() && i < new_value.mf_v_.size(); ++i) {
    value->mf_v_[i] += new_value.mf_v_[i];
  }

  value->wide_w_ += new_value.wide_w_;
  value->version_ = std::min(value->version_, new_value.version_);

  return ret;
}

int sparse_value_ver1_to_string(const SparseKeyVer1& key, const SparseValueVer1& value, std::string *str) {
  int ret = 0;
  (*str) = absl::StrFormat("%llu %u %d %llu %f %f %f",
           (unsigned long long)key,
           (unsigned int)value.slot_,
           (int)value.silent_days_,
           (unsigned long long)value.version_,
           value.delta_score_,
           value.show_,
           value.clk_);
  // LR model
  (*str) = (*str) + string(" ") + absl::StrFormat("%f %f",
           value.lr_w_,
           value.lr_g2sum_);
  // FM model
  (*str) = (*str) + string(" ") + absl::StrFormat("%f %f",
           value.fm_w_,
           value.fm_w_g2sum_);
  (*str) = (*str) + string(" ") + absl::StrFormat("%llu",
           (unsigned long long)value.fm_v_.size());
  for (size_t i = 0; i < value.fm_v_.size(); ++i) {
    (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.fm_v_[i]);
  }
  (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.fm_v_g2sum_);
  // MF model
  (*str) = (*str) + string(" ") + absl::StrFormat("%f %f",
           value.mf_w_,
           value.mf_w_g2sum_);
  (*str) = (*str) + string(" ") + absl::StrFormat("%llu",
           (unsigned long long)value.mf_v_.size());
  for (size_t i = 0; i < value.mf_v_.size(); ++i) {
    (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.mf_v_[i]);
  }
  (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.mf_v_g2sum_);
  // Wide
  (*str) = (*str) + string(" ") + absl::StrFormat("%f %f",
           value.wide_w_,
           value.wide_g2sum_);
  return ret;
}

int sparse_value_ver1_time_decay(SparseValueVer1 *value, const ps::runtime::TrainingRule& rule) {
  int ret = 0;
  ++(value->silent_days_);
  value->show_ *= rule.sparse_.cvm_rule_.decay_rate_;
  value->clk_  *= rule.sparse_.cvm_rule_.decay_rate_;
  return ret;
}

bool sparse_value_ver1_shrink(const SparseValueVer1& value, const ps::runtime::TrainingRule& rule) {
  float score = (value.show_ - value.clk_) * rule.sparse_.nonclk_coeff_ + value.clk_ * rule.sparse_.clk_coeff_;
  return (score < rule.sparse_.delete_threshold_) || (value.silent_days_ > rule.sparse_.delete_after_silent_days_);
}

} // namespace param_table
} // namespace ps

