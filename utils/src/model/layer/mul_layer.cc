#include "model/layer/mul_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void MultiplicationLayer::load_config(Config conf) {
  a_input_ = component_table()->load_component<MatrixOutput>(conf["a_input"]);
  b_input_ = component_table()->load_component<MatrixOutput>(conf["b_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  a_trans_ = conf["a_trans"].as<bool>();
  b_trans_ = conf["b_trans"].as<bool>();
  adding_output_ = conf["adding_output"].as<bool>();
}

void MultiplicationLayer::initialize() {
  CHECK(a_input_);
  CHECK(b_input_);
  CHECK(output_);
}

void MultiplicationLayer::feed_forward() {
  Eigen::MatrixXf& a_val = a_input_->value();
  Eigen::MatrixXf& b_val = b_input_->value();
  Eigen::MatrixXf& out_val = output_->value();

  if (adding_output_) {
    matrix_multiply_add(a_trans_, b_trans_, false, a_val, b_val, out_val);
  } else {
    matrix_multiply(a_trans_, b_trans_, false, a_val, b_val, out_val);
  }
  if (!adding_output_) {
    output_->need_gradient() = false;
    output_->has_gradient() = false;
  } else {
    CHECK(!output_->has_gradient());
  }

  if (a_input_->need_gradient() || b_input_->need_gradient()) {
    output_->need_gradient() = true;
  }
}

void MultiplicationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& a_val = a_input_->value();
    Eigen::MatrixXf& b_val = b_input_->value();
    Eigen::MatrixXf& out_grad = output_->gradient();
    if (a_input_->need_gradient()) {
      Eigen::MatrixXf& a_grad = a_input_->gradient();
      if (a_input_->has_gradient()) {
        matrix_multiply_add(false, !b_trans_, a_trans_, out_grad, b_val, a_grad);
      } else {
        matrix_multiply(false, !b_trans_, a_trans_, out_grad, b_val, a_grad);
      }
      a_input_->has_gradient() = true;
    }
    if (b_input_->need_gradient()) {
      Eigen::MatrixXf& b_grad = b_input_->gradient();
      if (b_input_->has_gradient()) {
        matrix_multiply_add(!a_trans_, false, b_trans_, a_val, out_grad, b_grad);
      } else {
        matrix_multiply(!a_trans_, false, b_trans_, a_val, out_grad, b_grad);
      }
      b_input_->has_gradient() = true;
    }
  }
}

} // namespace model
} // namespace ps

