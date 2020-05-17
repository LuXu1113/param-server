#include "model/layer/softmax_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void SoftMaxLayer::load_config(Config conf) {
  inputs_ = component_table()->load_components<MatrixOutput>(conf["input"]);
  outputs_ = component_table()->load_components<MatrixOutput>(conf["output"]);
}

void SoftMaxLayer::initialize() {
  //至少要2个以上才能做softmax
  CHECK(input_num() > 1 && output_num() == input_num());
  for (int i = 0; i < input_num(); i++) {
    CHECK(inputs_[i]);
  }
  for (int i = 0; i < output_num(); i++) {
    CHECK(outputs_[i]);
  }
}

void SoftMaxLayer::feed_forward() {
  // minibatch size
  int rown = inputs_[0]->value().rows();
  bool need_gradient = false;
  // 保存exp加和
  exp_sum_.setZero(rown, 1);

  for (int i = 0; i < input_num(); i++) {
    // all softmax input col must be 1
    CHECK(inputs_[i]->value().cols() == 1 && rown == inputs_[i]->value().rows());
    if (inputs_[i]->need_gradient()) {
      need_gradient = true;
    }

    Eigen::MatrixXf& in_val = inputs_[i]->value();
    exp_sum_ += (in_val.array().exp()).cast<float>();
  }

  for (int i = 0; i < output_num(); i++) {
    Eigen::MatrixXf& in_val = inputs_[i]->value();
    Eigen::MatrixXf& out_val = outputs_[i]->value();
    out_val = (in_val.array().exp()/exp_sum_).cast<float>();

    outputs_[i]->has_gradient() = false;
    outputs_[i]->need_gradient() = false;
    if (need_gradient == true) {
      outputs_[i]->need_gradient() = true;
    }
  }
}

void SoftMaxLayer::back_propagate() {
  for (int i = 0; i < input_num(); i++) {
    if (inputs_[i]->need_gradient()) {
      Eigen::MatrixXf& in_grad = inputs_[i]->gradient();
      Eigen::MatrixXf& out_i = outputs_[i]->value();
      //之前没有梯度，先setZero
      if (!inputs_[i]->has_gradient()) {
        in_grad.setZero(inputs_[i]->value().rows(), inputs_[i]->value().cols());
      }

      for (int j = 0; j < output_num(); j++) {
        // 加上回传梯度
        if (outputs_[j]->has_gradient()) {
          Eigen::MatrixXf& out_grad = outputs_[j]->gradient();
          Eigen::MatrixXf& out_j = outputs_[j]->value();
          if (i == j) {
            in_grad += (out_grad.array() * out_j.array() * (1-out_j.array())).matrix();
          } else {
            in_grad -= (out_grad.array() * out_i.array() * out_j.array()).matrix();
          }
        }
      }
      inputs_[i]->has_gradient() = true;
    }
  }
}

} // namespace model
} // namespace ps

