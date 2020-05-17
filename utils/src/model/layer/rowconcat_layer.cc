#include "model/layer/rowconcat_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void RowConcatenationLayer::load_config(Config conf) {
  inputs_ = component_table()->load_components<MatrixOutput>(conf["input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void RowConcatenationLayer::initialize() {
  CHECK(input_num() >= 1);
  for (int i = 0; i < input_num(); i++) {
    CHECK(inputs_[i]);
  }
  CHECK(output_);
}

void RowConcatenationLayer::feed_forward() {
  size_t coln = 0;
  size_t tot_rown = 0;
  output_->has_gradient() = false;
  output_->need_gradient() = false;
  for (int i = 0; i < input_num(); i++) {
    coln = inputs_[i]->value().cols();
    tot_rown += inputs_[i]->value().rows();
    if (inputs_[i]->need_gradient()) {
      output_->need_gradient() = true;
    }
  }
  Eigen::MatrixXf& out_val = output_->value();
  out_val.resize(tot_rown, coln);
  size_t offset = 0;
  for (int i = 0; i < input_num(); offset += inputs_[i++]->value().rows()) {
    Eigen::MatrixXf& in_val = inputs_[i]->value();
    CHECK((size_t)in_val.cols() == coln);
    out_val.block(offset, 0, in_val.rows(), coln) = in_val;
  }
  CHECK(offset == tot_rown);
}

void RowConcatenationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    CHECK(out_grad.size() == output_->value().size());
    size_t coln = out_grad.cols();
    size_t tot_rown = out_grad.rows();
    size_t offset = 0;
    for (int i = 0; i < input_num(); offset += inputs_[i++]->value().rows()) {
      if (!inputs_[i]->need_gradient()) {
        continue;
      }
      Eigen::MatrixXf& in_grad = inputs_[i]->gradient();
      if (inputs_[i]->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, inputs_[i]->value()));
        in_grad += out_grad.block(offset, 0, inputs_[i]->value().rows(), coln);
      } else {
        in_grad = out_grad.block(offset, 0, inputs_[i]->value().rows(), coln);
        inputs_[i]->has_gradient() = true;
      }
    }
    CHECK(offset == tot_rown);
  }
}

} // namespace model
} // namespace ps

