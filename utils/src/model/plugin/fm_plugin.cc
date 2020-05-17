#include "model/plugin/fm_plugin.h"

namespace ps {
namespace model {

void FMPlugin::initialize() {
  dim_   = ps::runtime::ConfigManager::pick_training_rule().sparse_.fm_rule_.dim_;
  CHECK(dim_ >= 0 && dim_ % 2 == 0);

  slots_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.fm_rule_.slots_;
  create_threshold_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.fm_rule_.create_threshold_;
  joint_train_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.fm_rule_.joint_train_;
}

void FMPlugin::feed_forward(ThreadLocalData *data) {
  for (int i = 0; i < data->batch_size_; ++i) {
    data->minibatch_[i].fm_v_sums_.assign(dim_, 0.0);
    float *v_sums = &(data->minibatch_[i].fm_v_sums_[0]);
    float sum_weight = 0.0;

    for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
      sum_weight += data->minibatch_[i].fea_pulls_[f].fm_w_;

      if (!(data->minibatch_[i].fea_pulls_[f].fm_v_.empty())) {
        CHECK((int)data->minibatch_[i].fea_pulls_[f].fm_v_.size() == dim_);
        const float *fm_v = &(data->minibatch_[i].fea_pulls_[f].fm_v_[0]);
        for (int j = 0; j < dim_; ++j) {
          v_sums[j] += fm_v[j];
          sum_weight -= fm_v[j] * fm_v[j] * 0.5;
        }
      }
    }
    for (int j = 0; j < dim_; ++j) {
      sum_weight += v_sums[j] * v_sums[j] * 0.5;
    }

    data->minibatch_[i].fm_output_ = sum_weight;
    data->minibatch_[i].fm_pred_ = 1.0 / (1.0 + exp(-sum_weight));
  }
}

void FMPlugin::back_propagate(ThreadLocalData *data) {
  if ((joint_train_) || (data->phase_ == TrainingPhase::UPDATING)) {
    for (int i = 0; i < data->batch_size_; ++i) {
      float *v_sums = &(data->minibatch_[i].fm_v_sums_[0]);
      float grad = data->minibatch_[i].label_ - data->minibatch_[i].fm_pred_;

      for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
        if (joint_train_) {
          data->minibatch_[i].fea_pushs_[f].fm_w_ += grad;
        } else {
          data->minibatch_[i].fea_pushs_[f].fm_w_ = grad;
        }

        if (data->minibatch_[i].fea_pulls_[f].fm_v_.empty()) {
          data->minibatch_[i].fea_pushs_[f].fm_v_.clear();
          DLOG(INFO) << "Empty fm feature, sign = " << data->minibatch_[i].feas_[i].sign_;
        } else {
          CHECK((int)(data->minibatch_[i].fea_pushs_[f].fm_v_.size()) == dim_);
          float *fm_v_g = &(data->minibatch_[i].fea_pushs_[f].fm_v_[0]);
          const float *fm_v = &(data->minibatch_[i].fea_pulls_[f].fm_v_[0]);

          for (int j = 0; j < dim_; ++j) {
            if (joint_train_) {
              fm_v_g[j] += grad * (v_sums[j] - fm_v[j]);
            } else {
              fm_v_g[j] = grad * (v_sums[j] - fm_v[j]);
            }
          }
        }
      }
    }
  }
}

} // namespace model
} // namespace ps

