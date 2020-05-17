#ifndef UTILS_INCLUDE_MODEL_LAYER_NEURAL_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_NEURAL_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"
#include "model/layer/common/activation_function.h"
#include "model/layer/activation_layer.h"
#include "model/layer/colconcat_layer.h"
#include "model/layer/rowconcat_layer.h"
#include "model/layer/mul_layer.h"

namespace ps {
namespace model {

// 实现神经网络层
class NeuralLayer : public Layer {
 public:
  void load_config(ps::toolkit::Config conf);

  int a_input_num() {
    return (int)a_inputs_.size();
  }

  std::shared_ptr<MatrixOutput>& a_input(int i) {
    return a_inputs_[i];
  }

  std::vector<std::shared_ptr<MatrixOutput> >& a_inputs() {
    return a_inputs_;
  }

  int b_input_num() {
    return (int)b_inputs_.size();
  }

  std::shared_ptr<MatrixOutput>& b_input(int i) {
    return b_inputs_[i];
  }

  std::vector<std::shared_ptr<MatrixOutput> >& b_inputs() {
    return b_inputs_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }
  std::shared_ptr<ActivationFunction>& activation_function() {
    return act_func_;
  }

  void initialize() override;
  void finalize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  std::vector<std::shared_ptr<MatrixOutput> > a_inputs_;
  std::vector<std::shared_ptr<MatrixOutput> > b_inputs_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
  std::shared_ptr<ActivationFunction> act_func_;

  std::shared_ptr<ColConcatenationLayer> col_concate_layer_;
  std::shared_ptr<RowConcatenationLayer> row_concate_layer_;
  std::shared_ptr<MultiplicationLayer> mul_layer_;
  std::shared_ptr<ActivationLayer> act_layer_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_NEURAL_LAYER_H_

