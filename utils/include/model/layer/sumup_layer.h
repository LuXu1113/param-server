#ifndef UTILS_INCLUDE_MODEL_LAYER_SUMUP_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_SUMUP_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

// 实现 多个个矩阵相加的 神经层
class SumUpLayer : public Layer {
 public:
  // realize multi matrix cwise add,  C = A1 + A2 +。。。。。。
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& input(int i) {
    return inputs_[i];
  }

  std::vector<std::shared_ptr<MatrixOutput> >& inputs(){
    return inputs_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  bool& adding_output() {
    return adding_output_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  std::vector<std::shared_ptr<MatrixOutput> > inputs_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
  bool adding_output_ = false;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_SUMUP_LAYER_H_

