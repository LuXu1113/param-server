#include "model/layer/norm_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void NormalizationLayer::load_config(Config conf) {
  input_ = component_table()->load_component<MatrixOutput>(conf["input"]);
  sum_input_ = component_table()->load_component<MatrixOutput>(conf["summary_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void NormalizationLayer::initialize() {
  CHECK(input_);
  CHECK(sum_input_);
  CHECK(output_);
}

void NormalizationLayer::feed_forward() {
  Eigen::MatrixXf& in_val = input_->value();
  Eigen::MatrixXf& sum_val = sum_input_->value();
  Eigen::MatrixXf& out_val = output_->value();

  // sum 矩阵的判断
  CHECK(matrix_size_equals(sum_val, 3, in_val.cols()));
  // 基于 sum矩阵计算 归一化参数
  means_ = sum_val.array().row(1) / sum_val.array().row(0);
  scales_ = (sum_val.array().row(0) / sum_val.array().row(2)).sqrt();
  // 计算归一化结果
  out_val = (in_val.array().rowwise() - means_.array()).rowwise() * scales_.array();

  output_->need_gradient() = input_->need_gradient();
  output_->has_gradient() = false;
}

void NormalizationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    if (input_->need_gradient()) {
      Eigen::MatrixXf& in_grad = input_->gradient();
      if (input_->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, out_grad));
        in_grad.array() += out_grad.array().rowwise() * scales_.array();
      } else {
        in_grad = out_grad.array().rowwise() * scales_.array();
      }
      input_->has_gradient() = true;
    }
  }
  if (sum_input_->need_gradient()) {
    Eigen::MatrixXf& in_val = input_->value();
    Eigen::MatrixXf& sum_grad = sum_input_->gradient();
    if (sum_input_->has_gradient()) {
      CHECK(matrix_size_equals(sum_grad, 3, in_val.cols()));
      sum_grad.array().row(0) += in_val.rows();
      sum_grad.array().row(1) += in_val.array().colwise().sum();
      sum_grad.array().row(2) += (in_val.array().rowwise() - means_.array()).square().colwise().sum();
    } else {
      sum_grad.resize(3, in_val.cols());
      sum_grad.array().row(0).setConstant(in_val.rows());
      sum_grad.array().row(1) = in_val.array().colwise().sum();
      sum_grad.array().row(2) = (in_val.array().rowwise() - means_.array()).square().colwise().sum();
    }
  }
  sum_input_->has_gradient() = true;
}

} // namespace model
} // namespace ps
