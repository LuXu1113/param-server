#include "model/layer/common/activation_function.h"

#include <math.h>
#include <vector>

using std::vector;

namespace ps {
namespace model {

void LinearActivationFunction::compute(size_t n, const float *ins, float *outs, float *grads) {
  for (size_t i = 0; i < n; ++i) {
    outs[i] = ins[i];
  }
  if (grads != NULL) {
    for (size_t i = 0; i < n; ++i) {
      grads[i] = 1.0;
    }
  }
}

void ReluActivationFunction::compute(size_t n, const float *ins, float *outs, float *grads) {
  for (size_t i = 0; i < n; ++i) {
    outs[i] = (ins[i] < 0) ? (0) : (ins[i]);
  }
  if (grads != NULL) {
    for (size_t i = 0; i < n; ++i) {
      grads[i] = (ins[i] < 0) ? (0) : (1);
    }
  }
}

void StanhActivationFunction::compute(size_t n, const float *ins, float *outs, float *grads) {
  const double A = 1.7159;
  const double B = 2.0 / 3.0;
  static thread_local vector<float> buffer;
  buffer.resize(n);
  for (size_t i = 0; i < n; ++i) {
    buffer[i] = ins[i] * B;
  }
  for (size_t i = 0; i < n; ++i) {
    outs[i] = (1 - exp(-2 * buffer[i])) / (1 + exp(-2 * buffer[i]));
  }
  if (grads != NULL) {
    for (size_t i = 0; i < n; ++i) {
      grads[i] = A * B * (1 - outs[i] * outs[i]);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    outs[i] *= A;
  }
}

void SigmoidActivationFunction::compute(size_t n, const float *ins, float *outs, float *grads) {
  for (size_t i = 0; i < n; ++i) {
    outs[i] = 1 / (1 + exp(-ins[i]));
  }
  if(grads != NULL) {
    for (size_t i = 0; i < n; ++i) {
      grads[i] =  (1 - outs[i]) * outs[i];
    }
  }
}

void TanhActivationFunction::compute(size_t n, const float *ins, float *outs, float *grads) {
  for (size_t i = 0; i < n; ++i) {
    outs[i] = (1 - exp(-2 * ins[i])) / (1 + exp(-2 * ins[i]));
  }
  if(grads != NULL) {
    for (size_t i = 0; i < n; ++i) {
      grads[i] = 1 - outs[i] * outs[i];
    }
  }
}

} // namespace model
} // namespace ps

