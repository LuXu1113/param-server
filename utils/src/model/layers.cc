#include "model/layers.h"
#include "toolkit/factory.h"

using ps::toolkit::Factory;

namespace ps {
namespace model {

static bool is_inited_ = false;

void LayerFactory::init() {
  if (is_inited_) {
    return;
  }
  is_inited_ = true;
  Factory<Component>& factory = global_component_factory();

  factory.add<LinearActivationFunction>("linear_act_func");
  factory.add<ReluActivationFunction>("relu_act_func");
  factory.add<StanhActivationFunction>("stanh_act_func");
  factory.add<SigmoidActivationFunction>("sigmoid_act_func");
  factory.add<TanhActivationFunction>("tanh_act_func");

  factory.add<ActivationLayer>("act_layer");
  factory.add<LinearLayer>("linear_layer");
  factory.add<RowConcatenationLayer>("row_concate_layer");
  factory.add<ColConcatenationLayer>("col_concate_layer");
  factory.add<MultiplicationLayer>("mul_layer");
  factory.add<NeuralLayer>("neural_layer");
  factory.add<NormalizationLayer>("normalization_layer");
  factory.add<BatchNormalizationLayer>("batch_normalization_layer");
  factory.add<ColSelectLayer>("col_select_layer");
  factory.add<EmbeddingSumLayer>("embedding_sum_layer");
  factory.add<ProductLayer>("product_layer");
  factory.add<OutProductLayer>("outproduct_layer");
  factory.add<AddLayer>("add_layer");
  factory.add<SumUpLayer>("sum_layer");
  factory.add<FusionLayer>("fusion_layer");
  factory.add<GaussianProbLayer>("gaussprob_layer");
  factory.add<SoftMaxLayer>("softmax_layer");
  factory.add<ColRepLayer>("col_rep_layer");
  factory.add<WeightSumPoolLayer>("weight_sum_pool_layer");
  factory.add<CalibrationLayer>("calibration_layer");
}

} // namespace model
} // namespace ps

