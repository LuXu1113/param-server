#include "model/layer/colselect_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void ColSelectLayer::load_config(Config conf) {
  input_ = component_table()->load_component<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  range_ = conf["range"].as<std::vector<int> >();
}

void ColSelectLayer::initialize() {
  CHECK(input_);
  CHECK(output_);
  CHECK(range_.size() == 2 && range_[0] >= 0 && range_[1] >= range_[0]);
}

void ColSelectLayer::feed_forward() {
  CHECK(range_[1] <= input_->value().cols());
  Eigen::MatrixXf& in_val = input_->value();
  Eigen::MatrixXf& out_val = output_->value();
  out_val = in_val.block(0, range_[0], input_->value().rows(), range_[1] - range_[0]);
  output_->need_gradient() = input_->need_gradient();
  output_->has_gradient() = false;
}

void ColSelectLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    CHECK(matrix_size_equals(out_grad, output_->value()));
    if (input_->need_gradient()) {
      Eigen::MatrixXf& in_grad = input_->gradient();
      if (input_->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, input_->value()));
        in_grad.block(0, range_[0], input_->value().rows(), range_[1] - range_[0]) += out_grad;
      } else {
        in_grad.setZero(input_->value().rows(), input_->value().cols());
        in_grad.block(0, range_[0], input_->value().rows(), range_[1] - range_[0]) = out_grad;
        input_->has_gradient() = true;
      }
    }
  }
}

} // namespace model
} // namespace ps

