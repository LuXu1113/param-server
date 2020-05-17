#include "model/plugin/lr_plugin.h"

namespace ps {
namespace model {

void LRPlugin::initialize() {
  joint_train_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.lr_rule_.joint_train_;
}

void LRPlugin::feed_forward(ThreadLocalData *data) {
  for (int i = 0; i < data->batch_size_; ++i) {
    float output = data->minibatch_[i].prior_;

    for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
      output += data->minibatch_[i].fea_pulls_[f].lr_w_;
    }

    data->minibatch_[i].lr_output_ = output;
    data->minibatch_[i].lr_pred_ = 1.0 / (1.0 + exp(-output));
  }
}

void LRPlugin::back_propagate(ThreadLocalData *data) {
  if ((joint_train_) || (data->phase_ == TrainingPhase::UPDATING)) {
    for (int i = 0; i < data->batch_size_; ++i) {
      float grad = data->minibatch_[i].label_ - data->minibatch_[i].lr_pred_;
      for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
        if (joint_train_) {
          data->minibatch_[i].fea_pushs_[f].lr_w_ += grad;
        } else {
          data->minibatch_[i].fea_pushs_[f].lr_w_ = grad;
        }
      }
    }
  }
}

} // namespace model
} // namespace ps

