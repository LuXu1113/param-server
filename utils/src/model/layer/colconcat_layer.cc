#include "model/layer/colconcat_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void ColConcatenationLayer::load_config(Config conf) {
  inputs_ = component_table()->load_components<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void ColConcatenationLayer::initialize() {
  CHECK(input_num() >= 1);
  for (int i = 0; i < input_num(); i++) {
    CHECK(inputs_[i]);
  }
  CHECK(output_);
}

void ColConcatenationLayer::feed_forward() {
  size_t rown = 0;
  size_t tot_coln = 0;
  output_->has_gradient() = false;
  output_->need_gradient() = false;
  for (int i = 0; i < input_num(); i++) {
    rown = inputs_[i]->value().rows();
    tot_coln += inputs_[i]->value().cols();
    if (inputs_[i]->need_gradient()) {
      output_->need_gradient() = true;
    }
  }

  Eigen::MatrixXf& out_val = output_->value();
  out_val.resize(rown, tot_coln);
  size_t offset = 0;
  for (int i = 0; i < input_num(); offset += inputs_[i++]->value().cols()) {
    Eigen::MatrixXf& in_val = inputs_[i]->value();
    CHECK((size_t)in_val.rows() == rown);
    out_val.block(0, offset, rown, in_val.cols()) = in_val;
  }
  CHECK(offset == tot_coln);
}

void ColConcatenationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    CHECK(out_grad.size() == output_->value().size());
    size_t rown = out_grad.rows();
    size_t tot_coln = out_grad.cols();
    size_t offset = 0;

    for (int i = 0; i < input_num(); offset += inputs_[i++]->value().cols()) {
      if (!inputs_[i]->need_gradient()) {
        continue;
      }
      Eigen::MatrixXf& in_grad = inputs_[i]->gradient();
      if (inputs_[i]->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, inputs_[i]->value()));
        in_grad += out_grad.block(0, offset, rown, inputs_[i]->value().cols());
      } else {
        in_grad = out_grad.block(0, offset, rown, inputs_[i]->value().cols());
        inputs_[i]->has_gradient() = true;
      }
    }
    CHECK(offset == tot_coln);
  }
}

} // namespace model
} // namespace ps

