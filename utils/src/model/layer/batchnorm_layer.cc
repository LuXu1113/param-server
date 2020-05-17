#include "model/layer/batchnorm_layer.h"

#include <butil/logging.h>
#include "model/tool/eigen_impl.h"

using ps::toolkit::Config;

namespace ps {
namespace model {

void BatchNormalizationLayer::load_config(Config conf) {
  input_ = component_table()->load_component<MatrixOutput>(conf["input"]);
  batch_norm_param_ = component_table()->load_component<MatrixOutput>(conf["batch_norm_param"]);
  sum_input_ = component_table()->load_component<MatrixOutput>(conf["summary_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  activation_fn_ = conf["activation_fn"].as<bool>();
}

void BatchNormalizationLayer::initialize() {
  CHECK(input_);
  CHECK(sum_input_);
  CHECK(output_);
  CHECK(batch_norm_param_);
}

void BatchNormalizationLayer::feed_forward() {
  Eigen::MatrixXf& in_val = input_->value();
  Eigen::MatrixXf& param = batch_norm_param_->value();
  Eigen::MatrixXf& sum_val = sum_input_->value();
  Eigen::MatrixXf& out_val = output_->value();
  CHECK(param.rows() == 2 && param.cols() == in_val.cols());
  alpha_ = param.row(0);
  beta_ = param.row(1);
  CHECK(matrix_size_equals(sum_val, 3, in_val.cols()));
  means_ = sum_val.array().row(1) / sum_val.array().row(0);
  scales_ = (sum_val.array().row(0) / sum_val.array().row(2)).sqrt();
  out_val = (in_val.array().rowwise() - means_.array()).rowwise() * scales_.array();
  out_val = out_val.array().rowwise() * alpha_.array();
  out_val.array().rowwise() += beta_.array();
  if (activation_fn_){
    out_val = out_val.cwiseMax(0);
  }
  output_->need_gradient() = input_->need_gradient();
  output_->has_gradient() = false;
}

void BatchNormalizationLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& out_grad = output_->gradient();
    Eigen::MatrixXf& out_val = output_->value();
    Eigen::MatrixXf out_acf_grad;
    // batch_norm 后是否加relu激活函数
    if (activation_fn_){
      out_acf_grad = (out_val.array() <=0).select(0, out_grad);
    } else {
      out_acf_grad = out_grad;
    }
    // inout 梯度计算
    if (input_->need_gradient()) {
      Eigen::MatrixXf& in_grad = input_->gradient();
      if (input_->has_gradient()) {
        CHECK(matrix_size_equals(in_grad, out_acf_grad));
        in_grad.array() += (out_acf_grad.array().rowwise() * scales_.array()).array().rowwise() * alpha_.array();
      } else {
        in_grad = (out_acf_grad.array().rowwise() * scales_.array()).array().rowwise() * alpha_.array();
      }
      input_->has_gradient() = true;
    }
    // batch_norm 梯度计算
    if (batch_norm_param_->need_gradient()) {
      Eigen::MatrixXf& param_grad = batch_norm_param_->gradient();
      param_grad.resize(2, out_acf_grad.cols());
      if (batch_norm_param_->has_gradient()) {
        //param_grad.row(0).array() += (out_acf_grad.array()*_out_norm.array()).array().colwise().sum();
        param_grad.row(0).array() += (out_acf_grad.array().rowwise() * scales_.array()).array().colwise().sum();
        param_grad.row(1).array() += out_acf_grad.array().colwise().sum();
      } else {
        //param_grad.row(0) = (out_acf_grad.array()*_out_norm.array()).array().colwise().sum();
        param_grad.row(0) = (out_acf_grad.array().rowwise() * scales_.array()).array().colwise().sum();
        param_grad.row(1) = out_acf_grad.array().colwise().sum();
      }
      batch_norm_param_->has_gradient() = true;
    }
  }
  // summary 参数更新
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

