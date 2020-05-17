#include "model/layer/neural_layer.h"

#include <butil/logging.h>

using ps::toolkit::Config;

namespace ps {
namespace model {

void NeuralLayer::load_config(Config conf) {
  a_inputs_ = component_table()->load_components<MatrixOutput>(conf["a_input"]);
  b_inputs_ = component_table()->load_components<MatrixOutput>(conf["b_input"]);
  output_ = component_table()->load_component<MatrixOutput>(conf["output"]);
  act_func_ = component_table()->load_component<ActivationFunction>(conf["act_func"]);
}

void NeuralLayer::initialize() {
  CHECK(!a_inputs_.empty());
  CHECK(!b_inputs_.empty());
  CHECK(output_);
  CHECK(act_func_);
  mul_layer_ = std::make_shared<MultiplicationLayer>();

  if (a_inputs_.size() == 1) {
    col_concate_layer_ = nullptr;
    mul_layer_->a_input() = a_inputs_[0];
  } else {
    col_concate_layer_ = std::make_shared<ColConcatenationLayer>();
    col_concate_layer_->inputs() = a_inputs_;
    col_concate_layer_->initialize();
    mul_layer_->a_input() = col_concate_layer_->output();
  }
  if (b_inputs_.size() == 1) {
    row_concate_layer_ = nullptr;
    mul_layer_->b_input() = b_inputs_[0];
  } else {
    row_concate_layer_ = std::make_shared<RowConcatenationLayer>();
    row_concate_layer_->inputs() = b_inputs_;
    row_concate_layer_->initialize();
    mul_layer_->b_input() = row_concate_layer_->output();
  }

  mul_layer_->initialize();
  act_layer_ = std::make_shared<ActivationLayer>();
  act_layer_->input() = mul_layer_->output();
  act_layer_->output() = output_;
  act_layer_->activation_function() = act_func_;
  act_layer_->initialize();
}

void NeuralLayer::finalize() {
  if (col_concate_layer_) {
    col_concate_layer_->finalize();
  }
  if (row_concate_layer_) {
    row_concate_layer_->finalize();
  }
  mul_layer_->finalize();
  act_layer_->finalize();
}

void NeuralLayer::feed_forward() {
  if (col_concate_layer_) {
    col_concate_layer_->feed_forward();
  }
  if (row_concate_layer_) {
    row_concate_layer_->feed_forward();
  }
  mul_layer_->feed_forward();
  act_layer_->feed_forward();
}

void NeuralLayer::back_propagate() {
  act_layer_->back_propagate();
  mul_layer_->back_propagate();
  if (row_concate_layer_) {
    row_concate_layer_->back_propagate();
  }
  if (col_concate_layer_) {
    col_concate_layer_->back_propagate();
  }
}

} // namespace model
} // namespace ps

