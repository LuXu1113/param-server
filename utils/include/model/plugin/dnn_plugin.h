#ifndef UTILS_INCLUDE_MODEL_PLUGIN_ELS_DNN_PLUGIN_H_
#define UTILS_INCLUDE_MODEL_PLUGIN_ELS_DNN_PLUGIN_H_

#include <math.h>
#include <memory>
#include <vector>
#include <string>
#include "param_table/data/sparse_kv_ver1.h"
#include "param_table/data/dense_value_ver1.h"
#include "param_table/data/summary_value_ver1.h"
#include "model/data/slot_array.h"
#include "model/data/instance.h"
#include "model/data/thread_local_data.h"
#include "model/plugin/common/plugin.h"

namespace ps {
namespace model {

class ELSDNNPlugin final : public Plugin {
 public:
  ELSDNNPlugin();

  void initialize() override;
  void finalize() override {}
  void load_model(const std::string& path) override {}
  void save_model(const std::string& path) override {}
  void preprocess(ThreadLocalData *data) override {}
  void feed_forward(ThreadLocalData *data) override;
  void back_propagate(ThreadLocalData *data) override;

  void build_graph(ThreadLocalData *data);
  void init_dnn_param(std::vector<ps::param_table::DenseValueVer1> *init_w);
  void init_summary_param(std::vector<ps::param_table::SummaryValueVer1> *init_s);
  inline uint64_t tot_param_len() const {
    return tot_param_len_;
  }
  inline uint64_t tot_summary_len() const {
    return tot_summary_len_;
  }

 private:
  void get_pull_dense(const std::vector<ps::param_table::DenseValueVer1Pull>& data, std::vector<std::shared_ptr<MatrixOutput> > *params);
  void set_push_dense(std::vector<ps::param_table::DenseValueVer1Push> *data, const std::vector<std::shared_ptr<MatrixOutput> >& params);
  void get_pull_summaries(const std::vector<ps::param_table::SummaryValueVer1>& data, std::vector<std::shared_ptr<MatrixOutput> > *summaries);
  void set_push_summaries(std::vector<ps::param_table::SummaryValueVer1> *data, const std::vector<std::shared_ptr<MatrixOutput> >& summaries);
  void sparse_feature_to_tensor(ThreadLocalData *data);
  void tensor_to_sparse_grad(ThreadLocalData *data);

  struct MyLayer {
    bool test_at_joining;
    bool train_at_joining;
    bool test_at_updating;
    bool train_at_updating;
    bool debug_at_test;
  };

  struct TensorInput {
    Eigen::MatrixXf show_mat_;
    Eigen::MatrixXf clk_mat_;
    Eigen::MatrixXf lr_mat_;
    Eigen::MatrixXf mf_mat_;
    Eigen::MatrixXf lr_grad_;
    Eigen::MatrixXf mf_grad_;
  };

  struct DNNParam {
    std::string name_;
    int rown_;
    int coln_;
    int length_;
    float init_range_;
  };

  struct DNNSummary {
    std::string name_;
    int length_;
  };

  bool is_inited_ = false;

  bool use_wide_ = false;

  int memory_output_num_ = 0;
  std::vector<std::string> memory_output_names_;

  bool use_gaussian_ = false;
  std::vector<std::string> gaussian_names_;
  int gaussian_num_ = 0;

  int q_num_ = 0;
  std::vector<std::string> q_names_;

  // dnn
  int base_slot_num_;
  SlotArray base_slot_mapping_;
  bool back_propagate_input_ = true;

  // memory dnn
  int memory_slot_num_;
  SlotArray memory_slot_mapping_;
  bool back_memory_input_ = true;

  // space time
  int addition_slot_num_;
  SlotArray addition_slot_mapping_;
  bool back_addition_input_ = true;

  int q_weight_num_ = 0;
  std::vector<float> q_weight_;

  int param_num_;
  std::vector<DNNParam> params_;
  uint64_t tot_param_len_;

  float summary_decay_rate_;
  float summary_squared_sum_epsilon_;
  int summary_num_;
  uint64_t tot_summary_len_;
  std::vector<DNNSummary> summaries_;

  int layer_num_;
  std::vector<MyLayer> layers_;
};

} // namespace utils
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_PLUGIN_ELS_DNN_PLUGIN_H_

