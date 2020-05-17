#ifndef UTILS_INCLUDE_MODEL_LAYER_ADD_LAYER_H_
#define UTILS_INCLUDE_MODEL_LAYER_ADD_LAYER_H_

#include <memory>
#include <Eigen>
#include "toolkit/config.h"
#include "model/data/matrix_output.h"
#include "model/layer/common/layer.h"

namespace ps {
namespace model {

class AddLayer : public Layer {
 public:
  // realize matrix cwise add,  C = A + B
  void load_config(ps::toolkit::Config conf);

  std::shared_ptr<MatrixOutput>& a_input() {
    return a_input_;
  }

  std::shared_ptr<MatrixOutput>& b_input() {
    return b_input_;
  }

  std::shared_ptr<MatrixOutput>& output() {
    return output_;
  }

  bool& adding_output() {
    return adding_output_;
  }

  void initialize() override;
  void feed_forward() override;
  void back_propagate() override;

private:
  std::shared_ptr<MatrixOutput> a_input_;
  std::shared_ptr<MatrixOutput> b_input_;
  std::shared_ptr<MatrixOutput> output_ = std::make_shared<MatrixOutput>();
  bool adding_output_ = false;
};

} // namespace modps
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYER_ADD_LAYER_H_

