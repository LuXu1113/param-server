#include "model/layer/linear_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void LinearLayer::load_config(Config conf) {
  inputs_ = component_table()->load_components<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void LinearLayer::initialize() {
  CHECK(input_num() >= 1);
  for (int i = 0; i < input_num(); i++) {
    CHECK(inputs_[i]);
  }
  CHECK(output_);
}

void LinearLayer::feed_forward() {
  Eigen::MatrixXf& out_val = output_->value();
  output_->has_gradient() = false;
  output_->need_gradient() = false;
  for (int i = 0; i < input_num(); i++) {
    Eigen::MatrixXf& in_val = inputs_[i]->value();
    if (i == 0) {
      out_val = in_val;
    } else {
      out_val += in_val;
    }
    if (inputs_[i]->need_gradient()) {
      output_->need_gradient() = true;
    }
  }
}

void LinearLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    for (int i = 0; i < input_num(); i++) {
      if (inputs_[i]->need_gradient()) {
        Eigen::MatrixXf& in_grad = inputs_[i]->gradient();
        if (inputs_[i]->has_gradient()) {
          in_grad += out_grad;
        } else {
          in_grad = out_grad;
          inputs_[i]->has_gradient() = true;
        }
      }
    }
  }
}

} // namespace model
} // namespace ps

