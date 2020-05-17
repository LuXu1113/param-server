#ifndef UTILS_INCLUDE_INCLUDE_CONFIG_MANAGER_H_
#define UTILS_INCLUDE_INCLUDE_CONFIG_MANAGER_H_

#include <vector>
#include <string>
#include "toolkit/config.h"
#include "toolkit/rpc_agent.h"

namespace ps {
namespace runtime {

struct ShardInfo {
  int global_id_;
  int local_id_;
  int mpi_rank_;
};

struct VecInput {
  std::string name_;
  int dim_;
};

struct AdditionInput {
  std::string name_;
  int dim_;
  int vlen_;
};

struct SummaryRule {
  std::string name_;
  int length_;
};

struct ParamRule {
  std::string name_;
  int row_n_;
  int col_n_;
  float init_range_;
  bool scale_by_row_n_;
};

struct LayerRule {
  std::string type_;
  std::string tag_;
  ps::toolkit::Config conf;
};

struct DenseTrainingRule {
  std::vector<std::string> q_names_;
  std::vector<float> q_weight_;

  bool use_gaussian_;
  std::vector<std::string> gaussian_output_names_;
  std::vector<std::string> memory_output_names_;

  bool use_wide_;
  bool back_propagate_input_;
  bool back_addition_input_;
  bool back_memory_input_;

  float summary_decay_rate_;
  float summary_squared_sum_epsilon_;
  float summary_init_n_;
  float summary_init_squared_sum_;

  float global_init_range_;

  float learning_rate_;
  float avg_decay_rate_;
  float ada_decay_rate_;
  float ada_epsilon_;
  float mom_decay_rate_;
  float weight_decay_;
  std::string optimizer_;

  std::vector<std::string> test_layers_at_joining_;
  std::vector<std::string> train_layers_at_joining_;
  std::vector<std::string> test_layers_at_updating_;
  std::vector<std::string> train_layers_at_updating_;

  std::vector<VecInput>      vec_input_;
  std::vector<AdditionInput> addition_input_;

  std::vector<SummaryRule> summary_;
  std::vector<ParamRule>   param_;
  std::vector<ps::toolkit::Config> layer_conf_;
  ps::toolkit::Config     all_layers_conf_;
};

struct CVMTrainingRule {
  bool  joint_train_;
  float decay_rate_;
};

struct LRTrainingRule {
  bool joint_train_;
  bool version_aware_;
  float learning_rate_;
  float initial_g2sum_;
  float initial_range_;
  float weight_upper_bound_;
  float weight_lower_bound_;
};

struct FMTrainingRule {
  bool joint_train_;
  bool version_aware_;
  std::vector<int> slots_;
  int dim_;
  float create_threshold_;
  float learning_rate_;
  float initial_g2sum_;
  float initial_range_;
  float weight_upper_bound_;
  float weight_lower_bound_;
};

struct MFTrainingRule {
  bool joint_train_;
  bool version_aware_;
  std::vector<int> slots_;
  int dim_;
  float create_threshold_;
  float learning_rate_;
  float initial_g2sum_;
  float initial_range_;
  float weight_upper_bound_;
  float weight_lower_bound_;
};

struct DicTrainingRule {
  bool version_aware_;
  int dim_;
  int delete_after_silent_days_;
  float create_threshold_;
  float delete_threshold_;
  float join_threshold_;
  float decay_rate_;
  float learning_rate_;
  float initial_range_;
  float mom_decay_rate_;
  float ada_decay_rate_;
  float ada_epsilon_;
  float initial_g2sum_;
  float weight_upper_bound_;
  float weight_lower_bound_;
};

struct WideTrainingRule {
  bool joint_train_;
  bool version_aware_;
  std::vector<int> slots_;
  float learning_rate_;
  float initial_g2sum_;
  float initial_range_;
  float weight_upper_bound_;
  float weight_lower_bound_;
};

struct SparseTrainingRule {
  bool use_quantized_embedding_;
  float create_clk_prob_;
  float create_nonclk_prob_;
  float clk_coeff_;
  float nonclk_coeff_;
  float join_threshold_;
  float delete_threshold_;
  int   delete_after_silent_days_;
  std::vector<int> base_slots_;
  std::vector<int> addition_slots_;
  std::vector<int> memory_slots_;

  CVMTrainingRule  cvm_rule_;
  LRTrainingRule   lr_rule_;
  FMTrainingRule   fm_rule_;
  MFTrainingRule   mf_rule_;
  DicTrainingRule  dic_rule_;
  WideTrainingRule wide_rule_;
};

struct TrainingRule {
  DenseTrainingRule  dense_;
  SparseTrainingRule sparse_;
};

struct DataShufflerRule {
  bool merge_by_lineid_;
  bool check_show_clk_consistency_;
  bool erase_duplicate_feas_;
  std::string delete_instance_with_out_slot_;
};

struct OfflineWorkerRule {
  bool shuffle_data_;
  bool test_per_step_;
  bool load_prior_model_;
  bool recover_mode_;
  std::string model_path_;
  std::string model_donefile_;
  std::string model_converter_;
  std::string data_path_;
  std::string data_donefile_;
  std::string data_converter_;

  int start_update_days_;
  std::string train_days_;
  std::string test_days_;
  std::string model_days_;

  bool dump_result_;
  std::string dump_result_path_;

  bool  open_sampling_;
  float sampling_rate_;

  std::string join_data_path_;
  bool backdate_;
  std::string backdate_days_;
};

struct OnlineWorkerRule {
  bool shuffle_data_;
  bool test_per_step_;
  bool load_prior_model_;
  bool recover_mode_;
  std::string model_path_;
  std::string model_donefile_;
  std::string model_converter_;
  std::string data_path_;
  std::string data_donefile_;
  std::string data_converter_;

  int start_update_days_;
  std::string train_days_;
  std::string test_days_;
  std::string model_days_;

  bool dump_result_;
  std::string dump_result_path_;

  bool  open_sampling_;
  float sampling_rate_;

  std::string join_data_path_;
  bool backdate_;
  std::string backdate_days_;

  float q_diff_left_threshold_;
  float q_diff_right_threshold_;
  float auc_diff_threshold_;
  int q_check_index_;

  std::string mq_reader_path_;
  int64_t data_block_size_;
  int test_step_interval_;
  int dump_model_interval_;
  int64_t fetch_data_timeout_;
  std::string mq_reader_name_;

  std::string batch_model_timestamp_;
  std::string start_dump_timestamp_;
  std::string dump_timestamp_;
  int64_t dump_time_interval_;
  int64_t dump_time_diff_;
  int64_t read_time_shift_;

  int max_null_fetch_iteration_;
  std::string mq_queue_name_;
  std::string mq_customize_options_;
  std::string user_identifier_;
  int log_print_interval_;
};

struct WorkerRule {
  int   batch_size_;
  bool  drop_feature_;
  bool  load_balance_;
  int bias_slot_;
  int position_slot_;
  std::vector<uint64_t> position_feas_;
  DataShufflerRule data_shuffler_rule_;
  std::string train_mode_;
  OfflineWorkerRule offline_worker_rule_;
  OnlineWorkerRule  online_worker_rule_;
};

class ConfigManager {
 public:
  ConfigManager() = delete;

  static void initialize(const std::string& config_file, bool is_worker);
  static void finalize();

  // server or client
  static void regist_is_worker(bool is_worker);
  static bool pick_is_worker();
  static void regist_is_inited(bool is_inited);
  static bool pick_is_inited();
  static void regist_rpc_server_info(const std::vector<ps::toolkit::RPCServerInfo>& rpc_server_info);
  static const std::vector<ps::toolkit::RPCServerInfo>& pick_rpc_server_info();

  // runtime config
  static void regist_local_thread_num(const int local_thread_num);
  static void regist_write_thread_num(const int write_thread_num);
  static void regist_disk_buffer_size(const size_t disk_buffer_size);
  static void regist_hdfs_buffer_size(const size_t hdfs_buffer_size);
  static void regist_hdfs_command(const std::string& hdfs_command);
  static void regist_hdfs_reader_command(const std::string& hdfs_reader_command);

  static int pick_local_thread_num();
  static int pick_write_thread_num();
  static size_t pick_disk_buffer_size();
  static size_t pick_hdfs_buffer_size();
  static const std::string& pick_hdfs_command();
  static const std::string& pick_hdfs_reader_command();
  static void regist_data_reader_default_capacity(size_t capacity);
  static void regist_data_reader_default_block_size(int block_size);
  static void regist_data_reader_default_thread_num(int thread_num);
  static size_t pick_data_reader_default_capacity();
  static int pick_data_reader_default_block_size();
  static int pick_data_reader_default_thread_num();

  // resource config
  static void regist_global_shard_num(const int global_shard_num);
  static void regist_local_shard_num(const int local_shard_num);
  static void regist_shard_info(const int mpi_size, const int mpi_rank);
  static const int pick_global_shard_num();
  static const int pick_local_shard_num();
  static const ShardInfo& pick_local_shard_info(const int local_shard_id);
  static const ShardInfo& pick_global_shard_info(const int global_shard_id);

  // model config
  static void regist_training_rule(const TrainingRule& rule);
  static const TrainingRule& pick_training_rule();

  // worker config
  static void regist_worker_rule(const WorkerRule& rule);
  static const WorkerRule& pick_worker_rule();

 private:
  static void load_framework_conf(ps::toolkit::Config& conf);
  static void load_plugins_conf(ps::toolkit::Config& conf);
  static void load_worker_conf(ps::toolkit::Config& conf);
};

} // namespace runtime
} // namespace ps

#endif // UTILS_INCLUDE_RUNTIME_CONFIG_MANAGER_H_

