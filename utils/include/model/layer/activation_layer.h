#ifndef UTILS_INCLUDE_MODEL_LAYER_ACTIVATION_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_ACTIVATION_LAYER_H_

#include <memory>
#include <Eigen>
#include <butil/logging.h>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/activation_function.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

// 实现神经元的激活层，包含 input 和 output, 都是矩阵， 以及一个 激活函数
class ActivationLayer : public Layer {
 public:
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& input() {
    return input_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  std::shared_ptr<ActivationFunction>& activation_function() {
    return act_func_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  std::shared_ptr<MatrixOutput> input_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
  std::shared_ptr<ActivationFunction> act_func_;
  Eigen::MatrixXf act_grad_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_ACTIVATION_LAYER_H_

