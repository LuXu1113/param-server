#include "model/plugin/cvm_plugin.h"

namespace ps {
namespace model {

void CVMPlugin::initialize() {
  decay_rate_  = ps::runtime::ConfigManager::pick_training_rule().sparse_.cvm_rule_.decay_rate_;
  joint_train_ = ps::runtime::ConfigManager::pick_training_rule().sparse_.cvm_rule_.joint_train_;
}

void CVMPlugin::back_propagate(ThreadLocalData *data) {
  if ((joint_train_) || (data->phase_ == TrainingPhase::UPDATING)) {
    for (int i = 0; i < data->batch_size_; ++i) {
      for (int f = 0; f < data->minibatch_[i].fea_num_; ++f) {
        data->minibatch_[i].fea_pushs_[f].show_ = 1;
        data->minibatch_[i].fea_pushs_[f].clk_  = data->minibatch_[i].label_;
      }
    }
  }
}

} // namespace model
} // namespace ps

