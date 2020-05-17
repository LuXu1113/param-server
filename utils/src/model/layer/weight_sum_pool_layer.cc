#include "model/layer/weight_sum_pool_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void WeightSumPoolLayer::load_config(Config conf) {
  input_len_ = conf["input_len"].as<int>();
  vec_len_ = conf["vec_len"].as<int>();
  a_input_ = component_table()->load_component<MatrixOutput>(conf["a_input"]);
  b_input_ = component_table()->load_component<MatrixOutput>(conf["b_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
}

void WeightSumPoolLayer::build_col_select_layer(std::shared_ptr<ColSelectLayer> col_select_layer, std::shared_ptr<MatrixOutput> input_all,
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
void WeightSumPoolLayer::build_product_layer(std::shared_ptr<ProductLayer> product_layer, std::shared_ptr<MatrixOutput> input_a,
                                             std::shared_ptr<MatrixOutput> input_b, std::shared_ptr<MatrixOutput> output) {
  product_layer->a_input() = input_a;
  product_layer->b_input() = input_b;
  // use outer output ?
  if (output != NULL) {
    product_layer->output() = output;
  }
  product_layer->initialize();
}

void WeightSumPoolLayer::initialize() {
  CHECK(output_);
  CHECK(a_input_);
  CHECK(b_input_);

  // prepare component we need
  for(int i = 0; i < vec_len_; i++) {
    // col_select for input_a
    a_input_select_layers_.push_back(std::make_shared<ColSelectLayer>());
    // col_select for input_b
    b_input_select_layers_.push_back(std::make_shared<ColSelectLayer>());
    // b_replicate
    b_replicate_layers_.push_back(std::make_shared<ColRepLayer>());
    // fusion_product
    fusion_product_layer_.push_back(std::make_shared<ProductLayer>());
  }
  // sum up layer
  sum_layer_ = std::make_shared<SumUpLayer>();
  sum_layer_->output() = output_;

  // build compute graph
  for (int i = 0; i < vec_len_; i++) {
    build_col_select_layer(a_input_select_layers_[i], a_input_, i*input_len_, (i+1)*input_len_);
    build_col_select_layer(b_input_select_layers_[i], b_input_, i, (i+1));

    b_replicate_layers_[i]->input() = b_input_select_layers_[i]->output();
    b_replicate_layers_[i]->dim() = input_len_;

    build_product_layer(fusion_product_layer_[i],  a_input_select_layers_[i]->output(), b_replicate_layers_[i]->output());
    sum_layer_->inputs().push_back(fusion_product_layer_[i]->output());
  }
}

void WeightSumPoolLayer::finalize() {
  for(int i = 0; i < vec_len_; i++) {
    // col_select for input_all
    a_input_select_layers_[i]->finalize();
    b_input_select_layers_[i]->finalize();
    b_replicate_layers_[i]->finalize();
    fusion_product_layer_[i]->finalize();
  }
  sum_layer_->finalize();
}

void WeightSumPoolLayer::feed_forward() {
  // check size
  CHECK(a_input_->value().rows() == b_input_->value().rows());
  CHECK(a_input_->value().cols() == input_len_ * b_input_->value().cols());
  CHECK(b_input_->value().cols() == vec_len_);

  for(int i = 0; i < vec_len_; i++) {
    // get input i
    a_input_select_layers_[i]->feed_forward();
    b_input_select_layers_[i]->feed_forward();
    b_replicate_layers_[i]->feed_forward();
    fusion_product_layer_[i]->feed_forward();
  }
  sum_layer_->feed_forward();
}

void WeightSumPoolLayer::back_propagate() {
  sum_layer_->back_propagate();

  for(int i = vec_len_ - 1; i >= 0; i--) {
    fusion_product_layer_[i]->back_propagate();
    b_replicate_layers_[i]->back_propagate();
    a_input_select_layers_[i]->back_propagate();
    b_input_select_layers_[i]->back_propagate();
  }
}


} // namespace model
} // namespace ps

