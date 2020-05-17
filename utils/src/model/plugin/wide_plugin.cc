#include "model/plugin/wide_plugin.h"

namespace ps {
namespace model {

void WidePlugin::initialize() {
  joint_train_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.wide_rule_.joint_train_;
}

void WidePlugin::feed_forward(ThreadLocalData *data) {
  for (int i = 0; i < data->batch_size_; ++i) {
    float output = data->minibatch_[i].prior_;
    for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
      output += data->minibatch_[i].fea_pulls_[f].wide_w_;
    }
    data->minibatch_[i].wide_output_ = output;
  }
}

void WidePlugin::back_propagate(ThreadLocalData *data) {
  if ((joint_train_) || (data->phase_ == TrainingPhase::UPDATING)) {
    if (joint_train_) {
      for (int i = 0; i < data->batch_size_; ++i) {
        for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
          data->minibatch_[i].fea_pushs_[f].wide_w_ = data->minibatch_[i].wide_grad_;
        }
      }
    }
  }
}

} // namespace model
} // namespace ps

