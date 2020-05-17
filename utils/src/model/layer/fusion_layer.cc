#include "model/layer/fusion_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void FusionLayer::load_config(Config conf) {
  input_len_ = conf["input_len"].as<int>();
  vec_len_ = conf["vec_len"].as<int>();
  a_input_ = component_table()->load_component<MatrixOutput>(conf["a_input"]);
  b_input_ = component_table()->load_component<MatrixOutput>(conf["b_input"]);
  bias_input_ = component_table()->load_component<MatrixOutput>(conf["bias_input"]);
  mask_input_ = component_table()->load_component<MatrixOutput>(conf["mask_input"]);
  param_ = component_table()->load_component<MatrixOutput>(conf["param"]);
  act_func_ =  component_table()->load_component<ActivationFunction>(conf["act_func"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void FusionLayer::initialize() {
  CHECK(output_);
  CHECK(a_input_);
  CHECK(b_input_);
  CHECK(bias_input_);
  CHECK(mask_input_);
  CHECK(param_);
  CHECK(act_func_);

  for(int i = 0; i < vec_len_; i++) {
    a_input_select_layers_.push_back(std::make_shared<ColSelectLayer>());
    mask_select_layers_.push_back(std::make_shared<ColSelectLayer>());
    fusion_weight_layers_.push_back(std::make_shared<NeuralLayer>());
    fusion_product_layers_.push_back(std::make_shared<ProductLayer>());
    mask_product_layers_.push_back(std::make_shared<ProductLayer>());
  }

  sum_layer_ = std::make_shared<SumUpLayer>();
  sum_layer_->output() = output_;

  for (int i = 0; i < vec_len_; i++) {
    build_col_select_layer(a_input_select_layers_[i], a_input_, i*input_len_, (i+1)*input_len_);
    build_col_select_layer(mask_select_layers_[i], mask_input_, i*input_len_, (i+1)*input_len_);

    std::vector<std::shared_ptr<MatrixOutput> > concate_inputs;
    concate_inputs.push_back(a_input_select_layers_[i]->output());
    concate_inputs.push_back(b_input_);
    concate_inputs.push_back(bias_input_);

    build_neural_layer(fusion_weight_layers_[i], concate_inputs, param_, act_func_);
    build_product_layer(fusion_product_layers_[i],  a_input_select_layers_[i]->output(), fusion_weight_layers_[i]->output());
    build_product_layer(mask_product_layers_[i], fusion_product_layers_[i]->output(), mask_select_layers_[i]->output());

    sum_layer_->inputs().push_back(mask_product_layers_[i]->output());
  }
}

void FusionLayer::finalize() {
  for(int i = 0; i < vec_len_; i++) {
    a_input_select_layers_[i]->finalize();
    mask_select_layers_[i]->finalize();

    fusion_weight_layers_[i]->finalize();
    fusion_product_layers_[i]->finalize();
    mask_product_layers_[i]->finalize();
  }
  sum_layer_->finalize();
}

void FusionLayer::feed_forward() {
  CHECK(a_input_->value().rows() == mask_input_->value().rows() && a_input_->value().rows() == b_input_->value().rows());
  CHECK(a_input_->value().cols() == mask_input_->value().cols() && a_input_->value().cols() == vec_len_ * b_input_->value().cols());

  for(int i = 0; i < vec_len_; i++) {
    a_input_select_layers_[i]->feed_forward();
    mask_select_layers_[i]->feed_forward();

    fusion_weight_layers_[i]->feed_forward();
    fusion_product_layers_[i]->feed_forward();

    mask_product_layers_[i]->feed_forward();
  }
  sum_layer_->feed_forward();
}

void FusionLayer::back_propagate() {
  sum_layer_->back_propagate();

  for(int i = vec_len_ - 1; i >= 0; i--) {
    mask_product_layers_[i]->back_propagate();
    fusion_product_layers_[i]->back_propagate();
    fusion_weight_layers_[i]->back_propagate();
    mask_select_layers_[i]->back_propagate();
    a_input_select_layers_[i]->back_propagate();
  }
}

void FusionLayer::build_neural_layer(std::shared_ptr<NeuralLayer> nerual_layer, const std::vector<std::shared_ptr<MatrixOutput>> & inputs,
                                     std::shared_ptr<MatrixOutput> weight, std::shared_ptr<ActivationFunction> act_func,
                                     std::shared_ptr<MatrixOutput> output) {
  for (int i = 0; i < (int)inputs.size(); i++) {
    nerual_layer->a_inputs().push_back(inputs[i]);
  }

  nerual_layer->b_inputs().push_back(weight);
  nerual_layer->activation_function() = act_func;
  if (output != NULL) {
    nerual_layer->output() = output;
  }
  nerual_layer->initialize();
}

void FusionLayer::build_col_select_layer(std::shared_ptr<ColSelectLayer> col_select_layer, std::shared_ptr<MatrixOutput> input_all,
                                         int start, int end, std::shared_ptr<MatrixOutput> output) {
  col_select_layer->input() = input_all;
  col_select_layer->range().push_back(start);
  col_select_layer->range().push_back(end);
  // use outer output ?
  if (output != NULL) {
    col_select_layer->output() = output;
  }
  col_select_layer->initialize();
}

// connect a product layer:
void FusionLayer::build_product_layer(std::shared_ptr<ProductLayer> product_layer, std::shared_ptr<MatrixOutput> input_a,
                                      std::shared_ptr<MatrixOutput> input_b, std::shared_ptr<MatrixOutput> output) {
  product_layer->a_input() = input_a;
  product_layer->b_input() = input_b;
  // use outer output ?
  if (output != NULL) {
    product_layer->output() = output;
  }
  product_layer->initialize();
}

} // namespace model
} // namespace ps

