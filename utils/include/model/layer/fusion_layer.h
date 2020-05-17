#ifndef UTILS_INCLUDE_MODEL_LAYER_FUSION_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_FUSION_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"
#include "model/layer/common/activation_function.h"
#include "model/layer/colselect_layer.h"
#include "model/layer/neural_layer.h"
#include "model/layer/product_layer.h"
#include "model/layer/sumup_layer.h"

namespace ps {
namespace model {

class FusionLayer : public Layer {
 public:
  /* example:
   *  - { class : fusion_layer, tag : only_join, input_len : 128, vec_len : 3, a_input : session_clk, b_input : local_ins,
   *              bias_input : bias, mask_input : mask, param : w, act_func: tanh, output : clk_fusion }
   */
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& a_input() {
    return a_input_;
  }

  std::shared_ptr<MatrixOutput>& b_input() {
    return b_input_;
  }

  std::shared_ptr<MatrixOutput>& bias_input() {
    return bias_input_;
  }

  std::shared_ptr<MatrixOutput>& mask_input() {
    return mask_input_;
  }

  std::shared_ptr<MatrixOutput>& param() {
    return param_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  std::shared_ptr<ActivationFunction>& act_function() {
    return act_func_;
  }

  void build_neural_layer(std::shared_ptr<NeuralLayer> nerual_layer, const std::vector<std::shared_ptr<MatrixOutput>> & inputs,
                          std::shared_ptr<MatrixOutput> weight, std::shared_ptr<ActivationFunction> act_func,
                          std::shared_ptr<MatrixOutput> output = NULL);
  void build_col_select_layer(std::shared_ptr<ColSelectLayer> col_select_layer, std::shared_ptr<MatrixOutput> input_all,
                          int start, int end, std::shared_ptr<MatrixOutput> output = NULL );
  void build_product_layer(std::shared_ptr<ProductLayer> product_layer, std::shared_ptr<MatrixOutput> input_a,
                          std::shared_ptr<MatrixOutput> input_b, std::shared_ptr<MatrixOutput> output = NULL );

  void initialize() override;
  void finalize() override;
  void feed_forward() override;
  void back_propagate() override;

 private:
  int input_len_;
  int vec_len_;
  std::shared_ptr<MatrixOutput> a_input_;
  std::shared_ptr<MatrixOutput> b_input_;
  std::shared_ptr<MatrixOutput> bias_input_;
  std::shared_ptr<MatrixOutput> mask_input_;
  std::shared_ptr<MatrixOutput> param_;
  std::shared_ptr<MatrixOutput> output_;
  std::shared_ptr<ActivationFunction> act_func_;

  std::vector<std::shared_ptr<ColSelectLayer> > a_input_select_layers_;
  std::vector<std::shared_ptr<ColSelectLayer> > mask_select_layers_;
  std::vector<std::shared_ptr<NeuralLayer> > fusion_weight_layers_;

  std::vector<std::shared_ptr<ProductLayer> > fusion_product_layers_;
  std::vector<std::shared_ptr<ProductLayer> > mask_product_layers_;
  std::shared_ptr<SumUpLayer> sum_layer_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_FUSION_LAYER_H_

