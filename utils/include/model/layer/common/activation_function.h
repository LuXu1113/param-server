#ifndef UTILS_INCLUDE_MODEL_LAYER_COMMON_ACTIVATION_FUNCTION_H_
#define UTILS_INCLUDE_MODEL_LAYER_COMMON_ACTIVATION_FUNCTION_H_

#include "model/data/component.h"

namespace ps {
namespace model {

// 激活函数的基类
class ActivationFunction : public Component {
 public:
  virtual void compute(size_t n, const float *ins, float *outs, float *grads) = 0;
};

// 线性激活函数
class LinearActivationFunction : public ActivationFunction {
 public:
  void compute(size_t n, const float *ins, float *outs, float *grads) override;
};

// Relu 激活函数
class ReluActivationFunction : public ActivationFunction {
 public:
  void compute(size_t n, const float *ins, float *outs, float *grads) override;
};

class StanhActivationFunction : public ActivationFunction {
 public:
  void compute(size_t n, const float *ins, float *outs, float *grads) override;
};

// sigmoid 激活函数
class SigmoidActivationFunction : public ActivationFunction {
 public:
  void compute(size_t n, const float *ins, float *outs, float *grads) override;
};

// tanh 激活函数
class TanhActivationFunction : public ActivationFunction {
 public:
  void compute(size_t n, const float *ins, float *outs, float *grads) override;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_COMMON_ACTIVATION_FUNCTION_H_

