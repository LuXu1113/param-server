#ifndef UTILS_INCLUDE_MODEL_DATA_THREAD_LOCAL__DATA_H_
#define UTILS_INCLUDE_MODEL_DATA_THREAD_LOCAL__DATA_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <Eigen>
#include "param_table/data/dense_value_ver1.h"
#include "param_table/data/summary_value_ver1.h"
#include "model/data/instance.h"
#include "model/data/matrix_output.h"
#include "model/layers.h"

namespace ps {
namespace model {

enum TrainingPhase {
  JOINING = 0,
  UPDATING,
  BACKDATING,
};

struct VectorInput {
  // space-time addition input: dim * vlen
  // mask 用来取出padding的影响
  std::string name_;
  int dim_;
  int vlen_;
  std::shared_ptr<MatrixOutput> maskinput_;
  std::shared_ptr<MatrixOutput> vinput_;
};

struct ThreadLocalData {
  int tid_ = -1;
  int batch_size_ = 0;
  TrainingPhase phase_;

  std::vector<Instance> minibatch_;

  std::vector<ps::param_table::DenseValueVer1Pull> dnn_pulls_;
  std::vector<ps::param_table::DenseValueVer1Push> dnn_pushs_;

  std::vector<ps::param_table::SummaryValueVer1> dnn_summary_pulls_;
  std::vector<ps::param_table::SummaryValueVer1> dnn_summary_pushs_;

  std::shared_ptr<MatrixOutput> dnn_bias_input_           = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_position_input_       = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_fixed_position_input_ = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_show_input_           = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_clk_input_            = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_ctr_input_            = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_lr_input_             = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_fm_input_             = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_mf_input_             = std::make_shared<MatrixOutput>();
  std::shared_ptr<MatrixOutput> dnn_memory_input_         = std::make_shared<MatrixOutput>();

  std::map<std::string, VectorInput> vec_inputs_;
  std::map<std::string, VectorInput> addition_inputs_;

  std::vector<std::shared_ptr<MatrixOutput> > dnn_params_;
  std::vector<std::shared_ptr<MatrixOutput> > dnn_summaries_;
  std::vector<std::shared_ptr<Layer> > dnn_layers_;

  std::vector<std::shared_ptr<MatrixOutput> > dnn_output_;          // dnn logits
  std::vector<std::shared_ptr<MatrixOutput> > memory_output_;
  std::vector<std::shared_ptr<MatrixOutput> > gaussian_output_;

  std::vector<Eigen::MatrixXf> dnn_preds_;                          // dnn predicts
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DATA_THREAD_LOCAL_DATA_H_

