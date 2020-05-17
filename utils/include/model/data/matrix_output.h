#ifndef UTILS_INCLUDE_MODEL_DATA_MATRIX_OUTPUT_H_
#define UTILS_INCLUDE_MODEL_DATA_MATRIX_OUTPUT_H_

#include <Eigen>
#include "model/data/component.h"

namespace ps {
namespace model {

// 封装矩阵对象
class MatrixOutput : public Component {
 public:
  Eigen::MatrixXf& value() {
    return value_;
  }
  Eigen::MatrixXf& gradient() {
    return gradient_;
  }
  bool& need_gradient() {
    return need_gradient_;
  }
  bool& has_gradient() {
    return has_gradient_;
  }

 private:
  bool need_gradient_ = false;
  bool has_gradient_ = false;
  Eigen::MatrixXf value_;
  Eigen::MatrixXf gradient_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DATA_MATRIX_OUTPUT_H_

