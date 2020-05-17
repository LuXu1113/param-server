#ifndef UTILS_INCLUDE_MODEL_LAYER_COMMON_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_COMMON_LAYER_H_

#include "model/data/component.h"

namespace ps {
namespace model {

class Layer : public Component {
 public:
  virtual void initialize() {
  }
  virtual void finalize() {
  }
  virtual void feed_forward() {
  }
  virtual void back_propagate() {
  }
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_COMMON_LAYER_H_

