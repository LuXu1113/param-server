#ifndef UTILS_INCLUDE_MODEL_LAYER_GAUSSION_PROB_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_GAUSSION_PROB_LAYER_H_

#include <memory>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

// 实现基于归一化之后的值计算高斯联合概率的layer
class GaussianProbLayer : public Layer {
 public:
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& input() {
    return input_;
  }
  std::shared_ptr<MatrixOutput>& summaryinput_() {
    return sum_input_;
  }
  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  float gauss_cdf(float x);
  float gauss_pdf(float x);
  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  std::shared_ptr<MatrixOutput> input_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> sum_input_;
  std::string prob_func_;
  Eigen::RowVectorXf means_, scales_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_GAUSSION_PROB_LAYER_H_

