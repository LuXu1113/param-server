#ifndef UTILS_INCLUDE_MODEL_LAYER_BATCHNORM_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_BATCHNORM_LAYER_H_

#include <memory>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

class BatchNormalizationLayer : public Layer {
 public:
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& input() {
    return input_;
  }
  std::shared_ptr<MatrixOutput>& batch_norm_param() {
    return batch_norm_param_;
  }
  std::shared_ptr<MatrixOutput>& summaryinput_() {
    return sum_input_;
  }
  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }
  bool& activation_fn() {
    return activation_fn_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

private:
  std::shared_ptr<MatrixOutput> input_;
  std::shared_ptr<MatrixOutput> batch_norm_param_;
  std::shared_ptr<MatrixOutput> sum_input_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();

  Eigen::RowVectorXf means_, scales_, alpha_, beta_;
  bool activation_fn_ = false;
};

} // namespace utils
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_BATCHNORM_LAYER_H_

