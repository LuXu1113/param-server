#include "model/layer/sumup_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void SumUpLayer::load_config(Config conf) {
  inputs_ = component_table()->load_components<MatrixOutput>(conf["inputs"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  adding_output_ = conf["adding_output"].as<bool>();
}

void SumUpLayer::initialize() {
  CHECK(!inputs_.empty());
  CHECK(inputs_.size() >=2);
  CHECK(output_);
}

void SumUpLayer::feed_forward() {
  Eigen::MatrixXf& val = inputs_[0]->value();
  Eigen::MatrixXf& out_val = output_->value();

  if (adding_output_) {
    out_val.noalias() += val;
  } else {
    out_val.noalias() = val;
  }

  for(int i = 1; i < (int)inputs_.size(); i++){
    Eigen::MatrixXf& temp_val = inputs_[i]->value();

    out_val.noalias() += temp_val;
  }

  if (!adding_output_) {
    output_->need_gradient() = false;
    output_->has_gradient() = false;
  } else {
    CHECK(!output_->has_gradient());
  }

  for(int i = 0; i < (int)inputs_.size(); i++){
    if(inputs_[i]->need_gradient())
      output_->need_gradient() = true;
  }
}

void SumUpLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    for(int i = 0; i < (int)inputs_.size(); i++){
      if(inputs_[i]->need_gradient()) {
        Eigen::MatrixXf& input_grad = inputs_[i]->gradient();
        if( inputs_[i]->has_gradient()) {
          input_grad.noalias() += out_grad;
        } else {
          input_grad.noalias() = out_grad;
        }
        inputs_[i]->has_gradient() = true;
      }
    }
  }
}

} // namespace model
} // namespace ps

