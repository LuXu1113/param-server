#ifndef UTILS_DNN_INCLUDE_MODEL_LAYER_WEIGHT_SUM_POOL_LAYER_H_
#define UTILS_DNN_INCLUDE_MODEL_LAYER_WEIGHT_SUM_POOL_LAYER_H_

#include <memory>
#include <vector>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"
#include "model/layer/colselect_layer.h"
#include "model/layer/colrep_layer.h"
#include "model/layer/product_layer.h"
#include "model/layer/sumup_layer.h"

namespace ps {
namespace model {

class WeightSumPoolLayer : public Layer {
 public:
  /* example:
   *   - { class : weightsumpool_layer, tag : only_join, input_len : 128, vec_len : 3, a_input : session_clk, b_input : session_clk_weight, output : clk_fusion }
   */
  void load_config(ps::toolkit::Config conf);

  int & input_len() {
    return input_len_;
  }

  int & vec_len() {
    return vec_len_;
  }

  std::shared_ptr<MatrixOutput>& a_input() {
    return a_input_;
  }

  std::shared_ptr<MatrixOutput>& b_input() {
    return b_input_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

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
  std::shared_ptr<MatrixOutput> output_;

  std::vector<std::shared_ptr<ColSelectLayer> > a_input_select_layers_;
  std::vector<std::shared_ptr<ColSelectLayer> > b_input_select_layers_;
  std::vector<std::shared_ptr<ColRepLayer> > b_replicate_layers_;

  std::vector<std::shared_ptr<ProductLayer> > fusion_product_layer_;
  std::shared_ptr<SumUpLayer> sum_layer_;
};

} // namespace model
} // namespace ps

#endif // UTILS_DNN_INCLUDE_MODEL_LAYER_WEIGHT_SUM_POOL_LAYER_H_

