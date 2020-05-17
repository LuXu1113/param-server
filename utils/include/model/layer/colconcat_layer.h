#ifndef UTILS_INCLUDE_MODEL_LAYER_COLCONCAT_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_COLCONCAT_LAYER_H_

#include <memory>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

// 实现两个 矩阵 列拼接的 神经层
class ColConcatenationLayer : public Layer {
 public:
  // 输入矩阵可以是一个vector
  void load_config(ps::toolkit::Config conf);

  int input_num() {
    return inputs_.size();
  }

  std::shared_ptr<MatrixOutput>& input(int i) {
    return inputs_[i];
  }

  std::vector<std::shared_ptr<MatrixOutput> >& inputs() {
    return inputs_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  std::vector<std::shared_ptr<MatrixOutput> > inputs_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_COLCONCAT_LAYER_H_

