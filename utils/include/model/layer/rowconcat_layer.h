#ifndef UTILS_INCLUDE_MODEL_LAYER_ROWCONCAT_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_ROWCONCAT_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

// 实现行拼接的 layer
class RowConcatenationLayer : public Layer {
 public:
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

#endif // UTILS_INCLUDE_MODEL_LAYER_ROWCONCAT_LAYER_H_

