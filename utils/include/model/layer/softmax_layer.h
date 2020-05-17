#ifndef UTILS_INCLUDE_MODEL_LAYER_SOFTMAX_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_SOFTMAX_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

class SoftMaxLayer : public Layer {
 public:
  // softmax神经网络单元，输入是 n 个 minibatch * 1 的 input , 输出 是 n个 minibatch * 1的 概率向量
  void load_config(ps::toolkit::Config conf);

  int input_num() {
    return inputs_.size();
  }
  int output_num() {
    return outputs_.size();
  }
  std::shared_ptr<MatrixOutput>& input(int i) {
    return inputs_[i];
  }
  std::vector<std::shared_ptr<MatrixOutput> >& inputs() {
    return inputs_;
  }
  std::shared_ptr<MatrixOutput>& output(int i) {
    return outputs_[i];
  }
  std::vector<std::shared_ptr<MatrixOutput> >& outputs() {
    return outputs_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override ;

 private:
  std::vector<std::shared_ptr<MatrixOutput> > inputs_;
  std::vector<std::shared_ptr<MatrixOutput> > outputs_;
  Eigen::ArrayXXf exp_sum_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_SOFTMAX_LAYER_H_

