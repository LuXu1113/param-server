#include "model/layer/add_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void AddLayer::load_config(Config conf) {
  a_input_ = component_table()->load_component<MatrixOutput>(conf["a_input"]);
  b_input_ = component_table()->load_component<MatrixOutput>(conf["b_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  adding_output_ = conf["adding_output"].as<bool>();
}

void AddLayer::initialize() {
  CHECK(a_input_);
  CHECK(b_input_);
  CHECK(output_);
}

void AddLayer::feed_forward() {
  Eigen::MatrixXf& a_val = a_input_->value();
  Eigen::MatrixXf& b_val = b_input_->value();
  Eigen::MatrixXf& out_val = output_->value();
  if (adding_output_) {
    out_val.noalias() +=  (a_val + b_val);
  } else {
    out_val.noalias() =  (a_val + b_val);
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

void AddLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    if (a_input_->need_gradient()) {
      Eigen::MatrixXf& a_grad = a_input_->gradient();
      if (a_input_->has_gradient()) {
        a_grad.noalias() += out_grad;
      } else {
        a_grad.noalias() = out_grad;
      }
      a_input_->has_gradient() = true;
    }
    if (b_input_->need_gradient()) {
      Eigen::MatrixXf& b_grad = b_input_->gradient();
      if (b_input_->has_gradient()) {
        b_grad.noalias() += out_grad;
      } else {
        b_grad.noalias() = out_grad;
      }
      b_input_->has_gradient() = true;
    }
  }
}

} // namespace model
} // namespace ps

