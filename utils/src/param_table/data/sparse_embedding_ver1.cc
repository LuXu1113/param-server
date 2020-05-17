#include "param_table/data/sparse_embedding_ver1.h"

#include <math.h>
#include <butil/logging.h>
#include "absl/strings/str_format.h"
#include "absl/random/random.h"

using std::string;
using ps::runtime::TrainingRule;
using ps::runtime::SparseTrainingRule;

namespace ps {
namespace param_table {

static inline void bound(float *x, const float lower_bound, const float upper_bound) {
  CHECK(!(lower_bound > upper_bound));
  if (*x < lower_bound) { *x = lower_bound;  }
  if (*x > upper_bound) { *x = upper_bound; }
}

SparseEmbeddingVer1 sparse_embedding_ver1_default() {
  SparseEmbeddingVer1 value;
  value.embedding_.clear();
  value.ada_d2sum_ = 0.0;
  value.ada_g2sum_.clear();
  value.slot_ = 0;
  value.silent_days_ = 0;
  value.count_ = 0.0;
  value.delta_score_ = 0.0;
  return value;
}

int sparse_embedding_ver1_init(SparseEmbeddingVer1 *value, const TrainingRule& rule) {
  static absl::BitGen gen;
  int ret = 0;
  const SparseTrainingRule& conf = rule.sparse_;

  value->embedding_.resize(conf.dic_rule_.dim_);
  for (int i = 0; i < int(value->embedding_.size()); ++i) {
    value->embedding_[i] = absl::uniform_real_distribution<float>(-conf.dic_rule_.initial_range_, conf.dic_rule_.initial_range_)(gen);
    bound(&(value->embedding_[i]), conf.dic_rule_.weight_lower_bound_, conf.dic_rule_.weight_upper_bound_);
  }
  value->ada_d2sum_ = 0.0;
  value->ada_g2sum_.assign(conf.dic_rule_.dim_, 0.0);

  value->slot_ = 0;
  value->silent_days_ = 0;
  value->count_ = 0.0;
  value->delta_score_ = 0.0;

  return ret;
}

int sparse_embedding_ver1_push(SparseEmbeddingVer1 *value, const SparseEmbeddingVer1& grad, const TrainingRule& rule) {
  int ret = 0;

  // CHECK(value->slot_ == grad.slot_) << "slot: " << value->slot_ << ", newslot: " << grad.slot_;

  const SparseTrainingRule& conf = rule.sparse_;
  value->count_ += grad.count_;
  value->delta_score_ += grad.count_;
  if ((grad.embedding_.size() != 0) && (value->embedding_.size() != 0)) {
    if (grad.count_ > 0) {
      float g_scale = grad.count_;
      if (conf.dic_rule_.version_aware_ && value->version_ > grad.version_) {
        g_scale *= sqrt(1.0 + (value->version_ - grad.version_));
      }
      value->ada_d2sum_ = conf.dic_rule_.ada_decay_rate_ * value->ada_d2sum_ + 1.0;

      for (int i = 0; i < conf.dic_rule_.dim_; ++i){
        float origin_grad = grad.embedding_[i];
        float scaled_grad = grad.embedding_[i] / g_scale;

        value->ada_g2sum_[i] = conf.dic_rule_.ada_decay_rate_ * value->ada_g2sum_[i] + scaled_grad * scaled_grad;
        float scale = sqrt((1.0 + conf.dic_rule_.ada_epsilon_) / (value->ada_g2sum_[i] / value->ada_d2sum_ + conf.dic_rule_.ada_epsilon_));

        value->embedding_[i] += conf.dic_rule_.learning_rate_ * origin_grad * scale;
        bound(&(value->embedding_[i]), conf.dic_rule_.weight_lower_bound_, conf.dic_rule_.weight_upper_bound_);
      }
    }
  }
  ++(value->version_);
  value->silent_days_ = 0;

  return ret;
}

int sparse_embedding_ver1_merge(SparseEmbeddingVer1 *value, const SparseEmbeddingVer1& new_value, const ps::runtime::TrainingRule& rule) {
  int ret = 0;
  // const SparseTrainingRule& conf = rule.sparse_;

  if (value->embedding_.size() < new_value.embedding_.size()) {
    value->embedding_.assign(0, new_value.embedding_.size());
  }
  for (size_t i = 0; i < value->embedding_.size() && i < new_value.embedding_.size(); ++i) {
    value->embedding_[i] += new_value.embedding_.size();
  }
  value->count_ += new_value.count_;
  value->version_ = std::min(value->version_, new_value.version_);

  return ret;
}

int sparse_embedding_ver1_to_string(const SparseKeyVer1& key, const SparseEmbeddingVer1& value, std::string *str) {
  int ret = 0;
  (*str) = absl::StrFormat("%llu %u %d %llu %f %f %f",
           (unsigned long long)key,
           (unsigned int)value.slot_,
           (int)value.silent_days_,
           (unsigned long long)value.version_,
           value.delta_score_,
           value.count_,
           value.ada_d2sum_);
  (*str) = (*str) + string(" ") + absl::StrFormat("%llu",
           (unsigned long long)value.embedding_.size());
  for (size_t i = 0; i < value.embedding_.size(); ++i) {
    (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.embedding_[i]);
  }
  (*str) = (*str) + string(" ") + absl::StrFormat("%llu",
           (unsigned long long)value.ada_g2sum_.size());
  for (size_t i = 0; i < value.ada_g2sum_.size(); ++i) {
    (*str) = (*str) + string(" ") + absl::StrFormat("%f", value.ada_g2sum_[i]);
  }
  return ret;
}

int sparse_embedding_ver1_time_decay(SparseEmbeddingVer1 *value, const ps::runtime::TrainingRule& rule) {
  int ret = 0;

  ++(value->silent_days_);
  value->count_ *= rule.sparse_.dic_rule_.decay_rate_;
  for (size_t i = 0; i < value->embedding_.size(); ++i) {
    value->embedding_[i] *= rule.sparse_.dic_rule_.decay_rate_;
  }

  return ret;
}

bool sparse_embedding_ver1_shrink(const SparseEmbeddingVer1& value, const ps::runtime::TrainingRule& rule) {
  return (value.count_ < rule.sparse_.dic_rule_.delete_threshold_) || (value.silent_days_ > rule.sparse_.dic_rule_.delete_after_silent_days_);
}

} // namespace param_table
} // namespace ps

