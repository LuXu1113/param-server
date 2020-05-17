#include "model/layer/colrep_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void ColRepLayer::load_config(Config conf) {
  input_ = component_table()->load_component<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  dim_ = conf["dim"].as<int>();
}

void ColRepLayer::initialize() {
  CHECK(input_);
  CHECK(output_);
  CHECK(dim_ > 0);
}

void ColRepLayer::feed_forward() {
  Eigen::MatrixXf& out_val = output_->value();
  Eigen::MatrixXf& in_val = input_->value();
  CHECK(input_->value().cols() == 1);
  out_val.setZero(input_->value().rows(), dim_);

  out_val = in_val.replicate(1, dim_);

  output_->need_gradient() = input_->need_gradient();
  output_->has_gradient() = false;
}

void ColRepLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    CHECK(matrix_size_equals(out_grad, output_->value()));
    if (input_->need_gradient()) {
      Eigen::MatrixXf& in_grad = input_->gradient();
      if (input_->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, input_->value()));
        in_grad += out_grad.rowwise().sum();
      } else {
        in_grad.setZero(input_->value().rows(), input_->value().cols());
        in_grad += out_grad.rowwise().sum();
        input_->has_gradient() = true;
      }
    }
  }
}

} // namespace model
} // namespace ps

