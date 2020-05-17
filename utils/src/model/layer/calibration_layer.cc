#include "model/layer/calibration_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void CalibrationLayer::load_config(Config conf) {
  input_  = component_table()->load_component<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  shift_  = conf["beta0_shift"].as<double>();
}

void CalibrationLayer::initialize() {
  CHECK(input_);
  CHECK(output_);
}

void CalibrationLayer::feed_forward() {
  Eigen::MatrixXf& out_val = output_->value();
  output_->has_gradient() = false;
  output_->need_gradient() = false;

  Eigen::MatrixXf& in_val = input_->value();
  out_val = in_val.array() + shift_;

  if (input_->need_gradient()) {
    output_->need_gradient() = true;
  }
}

void CalibrationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    Eigen::MatrixXf& in_grad = input_->gradient();
    if (input_->need_gradient()) {
      if (input_->has_gradient()) {
        in_grad += out_grad;
      } else {
        in_grad = out_grad;
      }
      input_->has_gradient() = true;
    }
  }
}

} // namespace model
} // namespace ps

