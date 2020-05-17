#ifndef UTILS_INCLUDE_MODEL_LAYERS_H_
#define UTILS_INCLUDE_MODEL_LAYERS_H_

#include "model/data/component.h"
#include "model/data/matrix_output.h"
#include "model/tool/eigen_impl.h"

#include "model/layer/common/activation_function.h"
#include "model/layer/common/layer.h"
#include "model/layer/activation_layer.h"
#include "model/layer/add_layer.h"
#include "model/layer/batchnorm_layer.h"
#include "model/layer/calibration_layer.h"
#include "model/layer/colconcat_layer.h"
#include "model/layer/colrep_layer.h"
#include "model/layer/colselect_layer.h"
#include "model/layer/embedding_sum_layer.h"
#include "model/layer/fusion_layer.h"
#include "model/layer/gaussion_prob_layer.h"
#include "model/layer/linear_layer.h"
#include "model/layer/mul_layer.h"
#include "model/layer/neural_layer.h"
#include "model/layer/norm_layer.h"
#include "model/layer/out_product_layer.h"
#include "model/layer/product_layer.h"
#include "model/layer/rowconcat_layer.h"
#include "model/layer/softmax_layer.h"
#include "model/layer/sumup_layer.h"
#include "model/layer/weight_sum_pool_layer.h"

namespace ps {
namespace model {

class LayerFactory {
 public:
  LayerFactory() = delete;
  static void init();
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_LAYERS_H_

