#include "model/layer/gaussion_prob_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void GaussianProbLayer::load_config(Config conf) {
  //输入 是一个 归一化处理之后的向量，每个元素服从高斯分布
  input_ = component_table()->load_component<MatrixOutput>(conf["input"]);

  // 输出是一个一维的值，高斯联合概率取log，然后再归一化，得到一个服从高斯分布的变量
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  sum_input_ = component_table()->load_component<MatrixOutput>(conf["summary_input"]);
  prob_func_ = conf["prob_func"].as<std::string>();
}

void GaussianProbLayer::initialize() {
  CHECK(input_);
  CHECK(sum_input_);
  CHECK(output_);
}

float GaussianProbLayer::gauss_cdf(float x) {
  const double a1 =  0.254829592;
  const double a2 = -0.284496736;
  const double a3 =  1.421413741;
  const double a4 = -1.453152027;
  const double a5 =  1.061405429;
  const double p  =  0.3275911;
  if(x < -10.0) x = -10.0;
  if(x > 10.0)  x = 10.0;
  int sign = 1;
  if(x < 0) sign = -1;
  x = fabs(x) / sqrt(2.0);
  double t = 1.0 / (1.0 + p * x);
  double y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  if(sign == 1){
    return 0.5 * (2.0 - y);
  } else {
    return 0.5 * y;
  }
}

float GaussianProbLayer::gauss_pdf(float x){
  if(x < -10.0) x = -10.0;
  if(x > 10.0) x = 10.0;
  double p = exp(-0.5 * x * x) / sqrt(2 * M_PI);
  return p;
}

void GaussianProbLayer::feed_forward() {
  Eigen::MatrixXf& out_val = output_->value();
  Eigen::MatrixXf& in_val = input_->value();
  out_val.setZero(in_val.rows(),1);
  //遍历行
  for (int i = 0; i < (int)in_val.rows(); i++) {
    for( int j = 0; j <  (int)in_val.cols(); j++) {
      if(prob_func_ == "cdf"){
        out_val(i,0) += log(gauss_cdf(in_val(i,j)));
      } else {
        out_val(i,0) += log(gauss_pdf(in_val(i,j)));
      }
    }
  }

  //进行归一化
  Eigen::MatrixXf& sum_val = sum_input_->value();
  means_ = sum_val.array().row(1) / sum_val.array().row(0);
  scales_ = (sum_val.array().row(0) / sum_val.array().row(2)).sqrt();

  //更新summary梯度
  Eigen::MatrixXf& sum_grad = sum_input_->gradient();
  sum_grad.setZero(3, out_val.cols());
  sum_grad.array().row(0) += out_val.rows();
  sum_grad.array().row(1) += out_val.array().colwise().sum();
  sum_grad.array().row(2) += (out_val.array().rowwise() - means_.array()).square().colwise().sum();
  sum_input_->has_gradient() = true;


  for(int i= 0; i< (int)out_val.rows(); i++){
    out_val(i,0) = (out_val(i, 0) - means_(0)) * scales_(0);
  }

  output_->need_gradient() = false;
  output_->has_gradient() = false;
}

void GaussianProbLayer::back_propagate() {
}

} // namespace model
} // namespace ps

