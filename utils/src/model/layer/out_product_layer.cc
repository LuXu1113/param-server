#include "model/layer/out_product_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void OutProductLayer::load_config(Config conf) {
  a_input_ = component_table()->load_component<MatrixOutput>(conf["a_input"]);
  b_input_ = component_table()->load_component<MatrixOutput>(conf["b_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  a_trans_ = conf["a_trans"].as<bool>();
  b_trans_ = conf["b_trans"].as<bool>();
  adding_output_ = conf["adding_output"].as<bool>();
}

void OutProductLayer::initialize() {
  CHECK(a_input_);
  CHECK(b_input_);
  CHECK(output_);
}

void OutProductLayer::feed_forward() {
  Eigen::MatrixXf& a_val = a_input_->value();
  Eigen::MatrixXf& b_val = b_input_->value();
  Eigen::MatrixXf& out_val = output_->value();
  CHECK(a_val.rows() == b_val.rows());

  out_val.resize(a_val.rows(), a_val.cols()*b_val.cols());
  Eigen::MatrixXf output__tmp = Eigen::MatrixXf::Zero(a_val.cols(), b_val.cols());
  for(int i = 0; i < a_val.rows(); i++) {
    if (adding_output_) {
      output__tmp += a_val.row(i).transpose() * b_val.row(i);
    } else {
      output__tmp = a_val.row(i).transpose() * b_val.row(i);
    }
    output__tmp.resize(1, a_val.cols() * b_val.cols());
    out_val.row(i) = output__tmp;
    output__tmp.resize(a_val.cols(), b_val.cols());
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

void OutProductLayer::back_propagate() {
  if (output_->has_gradient()) {
    Eigen::MatrixXf& a_val = a_input_->value();
    Eigen::MatrixXf& b_val = b_input_->value();
    Eigen::MatrixXf& out_grad = output_->gradient();
    Eigen::MatrixXf out_grad_block_a = Eigen::MatrixXf::Zero(a_val.rows(), a_val.cols());
    Eigen::MatrixXf out_grad_block_b = Eigen::MatrixXf::Zero(b_val.rows(), b_val.cols());
    Eigen::MatrixXf _grad_tmp = Eigen::MatrixXf::Zero(1, a_val.cols()*b_val.cols());
    for(int i = 0; i < out_grad.rows(); i++){
      _grad_tmp = out_grad.row(i);
      _grad_tmp.resize(a_val.cols(), b_val.cols());
      out_grad_block_a.row(i) = b_val.row(i) * _grad_tmp.transpose();
      out_grad_block_b.row(i) = a_val.row(i) * _grad_tmp;
    }
    if (a_input_->need_gradient()) {
      Eigen::MatrixXf& a_grad = a_input_->gradient();
      if (a_input_->has_gradient()) {
        a_grad.noalias() += out_grad_block_a;
        //printf("%d,%d",out_grad_blok.cols(),b_val.rows())
      } else {
        a_grad.noalias() = out_grad_block_a;
      }
      a_input_->has_gradient() = true;
    }
    if (b_input_->need_gradient()) {
      Eigen::MatrixXf& b_grad = b_input_->gradient();
      if (b_input_->has_gradient()) {
        b_grad.noalias() += out_grad_block_b;
      } else {
        b_grad.noalias() = out_grad_block_b;
      }
      b_input_->has_gradient() = true;
    }
  }
}

} // namespace model
} // namespace ps

