#ifndef UTILS_INCLUDE_MODEL_DISTRIBUTED_LEARNER_RTSPARSE_LEARNER_H_
#define UTILS_INCLUDE_MODEL_DISTRIBUTED_LEARNER_RTSPARSE_LEARNER_H_

#include <vector>

#include "param_table/dense_value_ver1_table.h"
#include "param_table/summary_value_ver1_table.h"
#include "param_table/sparse_kv_ver1_table.h"
#include "param_table/sparse_embedding_ver1_table.h"

#include "toolkit/channel.h"
#include "toolkit/operating_log.h"
#include "model/data/instance.h"
#include "model/data/thread_local_data.h"
#include "model/data/record.h"
#include "model/tool/auc_calculator.h"
#include "model/plugins.h"

namespace ps {
namespace model {

class RTSparseLearner {
 public:
  RTSparseLearner() = default;
  RTSparseLearner(const RTSparseLearner&) = delete;
  ~RTSparseLearner() = default;

  void initialize();
  void finalize();

  void begin_day();
  void end_day();

  void time_decay();
  void shrink_table();
  void begin_pass(TrainingPhase phase);
  void end_pass();
  void set_testmode(bool mode);

  // void load_model(const std::string& path, const std::string& converter);    // not implement
  // void save_model(const std::string& path, const std::string& converter);    // not implement
  // void write_done_file(const std::string&path, const std::string& done_str); // not implement

  // realtime sparse-leaner▒▒▒▒▒▒▒
  void process_data(ps::toolkit::Channel<Record> in_chan);
  void save_param_table(const std::string& path);

 private:
  // plugins
  CVMPlugin cvm_plugin_;
  LRPlugin lr_plugin_;
  MFPlugin mf_plugin_;
  FMPlugin fm_plugin_;
  ELSDNNPlugin ps_dnn_plugin_;
  WidePlugin wide_plugin_;

  // param tables
  ps::param_table::DenseValueVer1TableClient      dense_table_client_;
  ps::param_table::SummaryValueVer1TableClient    summary_table_client_;
  ps::param_table::SparseKVVer1TableClient        sparse_table_client_;
  ps::param_table::SparseEmbeddingVer1TableClient memory_table_client_;

  // -----
  bool use_sync_comm_ = false;
  bool is_initialized_ = false;
  bool dump_result_ = false;

  // threads
  int thread_num_;
  std::vector<ThreadLocalData> thread_local_data_;

  // -----
  int batch_size_;
  float create_clk_prob_;
  float create_nonclk_prob_;
  float delete_threshold_;
  int delete_after_silent_days_;
  SlotArray base_slot_set_;
  SlotArray memory_slot_set_;

  // -----
  std::vector<std::string> output_model_;

  // -----
  TrainingPhase phase_ = TrainingPhase::JOINING;
  bool test_mode_ = false;

  // auc calculator
  AUCCalculator lr_auc_;
  AUCCalculator mf_auc_;
  AUCCalculator fm_auc_;
  AUCCalculator adq_auc_;
  AUCCalculator *dnn_auc_;
  int dnn_auc_num_;

  // trace performance
  ps::toolkit::OperatingLog perf_preprocess_;
  ps::toolkit::OperatingLog perf_pull_;
  ps::toolkit::OperatingLog perf_push_;
  ps::toolkit::OperatingLog perf_forward_;
  ps::toolkit::OperatingLog perf_back_propagate_;
  ps::toolkit::OperatingLog perf_auc_;
  ps::toolkit::OperatingLog perf_pull_sparse_;
  ps::toolkit::OperatingLog perf_pull_dense_;
  ps::toolkit::OperatingLog perf_push_sparse_;
  ps::toolkit::OperatingLog perf_push_dense_;

  // internal initialize and finalize
  void initialize_param_table();
  void finalize_param_table();
  void initialize_thread_local_data();
  void finalize_thread_local_data();

  // training stages
  void pull_params(ThreadLocalData *data);
  void preprocess(ThreadLocalData *data);
  void feed_forward(ThreadLocalData *data);
  void back_propagate(ThreadLocalData *data);
  void push_params(ThreadLocalData *data);

  // local tools
  void init_pushs(ThreadLocalData *data);
  void record_to_instance(const Record& rec, Instance *ins);
  void process_data_thread(int tid, ps::toolkit::Channel<Record> in_chan);
};

} // namespace utils
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DISTRIBUTED_LEARNER_RTSPARSE_LEARNER_H_

