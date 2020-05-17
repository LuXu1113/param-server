#include "model/plugin/mf_plugin.h"

namespace ps {
namespace model {

void MFPlugin::initialize() {
  slots_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.mf_rule_.slots_;
  dim_   = ps::runtime::ConfigManager::pick_training_rule().sparse_.mf_rule_.dim_;
  CHECK(dim_ >= 0 && dim_ % 2 == 0);
  halfdim_ = dim_ / 2;

  create_threshold_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.mf_rule_.create_threshold_;
  joint_train_      = ps::runtime::ConfigManager::pick_training_rule().sparse_.mf_rule_.joint_train_;
}

void MFPlugin::feed_forward(ThreadLocalData *data) {
  for (int i = 0; i < data->batch_size_; ++i) {
    data->minibatch_[i].mf_v_sums_.assign(dim_, 0.0);
    float *v_sums = &(data->minibatch_[i].mf_v_sums_[0]);
    float sum_weight = 0.0;

    for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
      sum_weight += data->minibatch_[i].fea_pulls_[f].mf_w_;
      if (!(data->minibatch_[i].fea_pulls_[f].mf_v_.empty())) {
        CHECK((int)data->minibatch_[i].fea_pulls_[f].mf_v_.size() == dim_);
        const float *mf_v = &(data->minibatch_[i].fea_pulls_[f].mf_v_[0]);
        for (int j = 0; j < dim_; ++j) {
          v_sums[j] += mf_v[j];
        }
      }
    }

    for (int j = 0; j < halfdim_; ++j) {
      sum_weight += v_sums[j] * v_sums[halfdim_ + j];
    }

    data->minibatch_[i].mf_output_ = sum_weight;
    data->minibatch_[i].mf_pred_ = 1.0 / (1.0 + exp(-sum_weight));
  }
}

void MFPlugin::back_propagate(ThreadLocalData *data) {
  if ((joint_train_) || (data->phase_ == TrainingPhase::UPDATING)) {
    for (int i = 0; i < data->batch_size_; ++i) {
      float *v_sums = &(data->minibatch_[i].mf_v_sums_[0]);
      float grad = data->minibatch_[i].label_ - data->minibatch_[i].mf_pred_;

      for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
        if (joint_train_) {
          data->minibatch_[i].fea_pushs_[f].mf_w_ += grad;
        } else {
          data->minibatch_[i].fea_pushs_[f].mf_w_ = grad;
        }

        if (data->minibatch_[i].fea_pulls_[f].mf_v_.empty()) {
          data->minibatch_[i].fea_pushs_[f].mf_v_.clear();
          DLOG(INFO) << "Empty mf feature, sign = " << data->minibatch_[i].feas_[i].sign_;
        } else {
          CHECK((int)(data->minibatch_[i].fea_pushs_[f].mf_v_.size()) == dim_);
          float *mf_v_g = &(data->minibatch_[i].fea_pushs_[f].mf_v_[0]);

          for (int j = 0; j < halfdim_; ++j) {
            if (joint_train_){
              mf_v_g[j] += grad * v_sums[halfdim_ + j];
              mf_v_g[halfdim_ + j] += grad * v_sums[j];
            } else{
              mf_v_g[j] = grad * v_sums[halfdim_ + j];
              mf_v_g[halfdim_ + j] = grad * v_sums[j];
            }
          }
        }
      }
    }
  }
}

} // namespace model
} // namespace ps

