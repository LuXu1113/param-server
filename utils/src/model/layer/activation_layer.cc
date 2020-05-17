#include "model/layer/activation_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void ActivationLayer::load_config(Config conf) {
  input_    = component_table()->load_component<MatrixOutput>(conf["input"]);
  output_   = component_table()->load_component<MatrixOutput>(conf["output"]);
  act_func_ = component_table()->load_component<ActivationFunction>(conf["act_func"]);
}

void ActivationLayer::initialize() {
  CHECK(input_);
  CHECK(output_);
  CHECK(act_func_);
}

void ActivationLayer::feed_forward() {
  Eigen::MatrixXf& in_val  = input_->value();
  Eigen::MatrixXf& out_val = output_->value();
  out_val.resizeLike(in_val);

  if (input_->need_gradient()) {
    act_grad_.resizeLike(in_val);
    act_func_->compute(in_val.size(), in_val.data(), out_val.data(), act_grad_.data());
  } else {
    act_func_->compute(in_val.size(), in_val.data(), out_val.data(), NULL);
  }

  output_->has_gradient() = false;
  output_->need_gradient() = input_->need_gradient();
}

void ActivationLayer::back_propagate() {
  if (input_->need_gradient() && output_->has_gradient()) {
    Eigen::MatrixXf& in_grad = input_->gradient();
    Eigen::MatrixXf& out_grad = output_->gradient();
    if (input_->has_gradient()) {
      CHECK(matrix_size_equals(in_grad, out_grad) && matrix_size_equals(act_grad_, out_grad));
      in_grad += out_grad.cwiseProduct(act_grad_);
    } else {
      in_grad = out_grad.cwiseProduct(act_grad_);
    }
    input_->has_gradient() = true;
  }
}

} // namespace model
} // namespace ps

