#include "runtime/config_manager.h"

#include <stdlib.h>
#include <string>
#include <butil/logging.h>
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"
#include "toolkit/config.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/rpc_agent.h"
#include "toolkit/shell_agent.h"

using std::string;
using std::vector;
using ps::toolkit::ShellAgent;
using ps::toolkit::MPIAgent;
using ps::toolkit::RPCServerInfo;
using ps::toolkit::Config;

namespace ps {
namespace runtime {

static bool is_inited_ = false;
static bool is_worker_ = false;
static vector<struct RPCServerInfo> rpc_server_info_;

static struct RuntimeConfig {
  int    local_thread_num_    = 0;
  int    write_thread_num_    = 0;
  size_t disk_buffer_size_    = 0;
  size_t hdfs_buffer_size_    = 0;
  string hdfs_command_        = "";
  string hdfs_reader_command_ = "";
  size_t data_reader_default_capacity_   = 1000000;
  int data_reader_default_block_size_ = 8192;
  int data_reader_default_thread_num_ = 1;
} runtime_config_;

static struct ResourceConfig {
  int global_shard_num_    = 0;
  int local_shard_num_     = 0;
  vector<ShardInfo> global_shard_info_;
  vector<ShardInfo> local_shard_info_;
} resource_config_;

static struct TrainingRule training_rule_;
static struct WorkerRule   worker_rule_;

void ConfigManager::load_framework_conf(Config& conf) {
  if (conf["framework"].is_scalar()) {
    *conf["framework"] = YAML::LoadFile(conf["framework"].as<std::string>());
  }
  // shell env config
  Config env_var_conf = conf["framework"]["environment_variables"];
  PCHECK(env_var_conf.is_map());
  for (YAML::const_iterator it = env_var_conf->begin(); it != env_var_conf->end(); ++it) {
    string value = ShellAgent::shell_get_command_output("echo -n " + it->second.as<string>());
    LOG(INFO) << "setenv: " << it->first.as<string>().c_str() << " = " << value.c_str();
    PCHECK(0 == setenv(it->first.as<string>().c_str(), value.c_str(), 1));
  }

  // runtime config
  regist_local_thread_num(conf["framework"]["thread_num"].as<int>());
  regist_write_thread_num(conf["framework"]["write_thread_num"].as<int>());
  regist_disk_buffer_size(conf["framework"]["localfs_buffer_size"].as<size_t>());
  regist_hdfs_buffer_size(conf["framework"]["hdfs_buffer_size"].as<size_t>());
  regist_hdfs_command(conf["framework"]["hdfs_command"].as<string>());
  if (conf["framework"]["hdfs_reader_command"].is_defined()) {
    regist_hdfs_reader_command(conf["framework"]["hdfs_reader_command"].as<string>());
  }
  regist_data_reader_default_capacity(conf["framework"]["read_from_default_capacity"].as<size_t>());
  regist_data_reader_default_block_size(conf["framework"]["read_from_default_block_size"].as<int>());
  regist_data_reader_default_thread_num(conf["framework"]["read_from_default_thread_num"].as<int>());

  // resources config
  regist_local_shard_num(conf["framework"]["param_table"]["local_shard_num"].as<int>());
  regist_shard_info(MPIAgent::mpi_size_group(), MPIAgent::mpi_rank_group());
}

void ConfigManager::load_plugins_conf(Config& conf) {
  if (conf["plugins"].is_scalar()) {
    *conf["plugins"] = YAML::LoadFile(conf["plugins"].as<std::string>());
  }

  training_rule_.sparse_.create_clk_prob_          = conf["plugins"]["create_clk_prob"].as<float>();
  training_rule_.sparse_.create_nonclk_prob_       = conf["plugins"]["create_nonclk_prob"].as<float>();
  training_rule_.sparse_.clk_coeff_                = conf["plugins"]["clk_coeff"].as<float>();
  training_rule_.sparse_.nonclk_coeff_             = conf["plugins"]["nonclk_coeff"].as<float>();
  training_rule_.sparse_.join_threshold_           = conf["plugins"]["join_threshold"].as<float>();
  training_rule_.sparse_.delete_threshold_         = conf["plugins"]["delete_threshold"].as<float>();
  training_rule_.sparse_.delete_after_silent_days_ = conf["plugins"]["delete_after_silent_days"].as<int>();
  if (conf["plugins"]["use_quantization_embedding"].is_defined()) {
    training_rule_.sparse_.use_quantized_embedding_ = conf["plugins"]["use_quantization_embedding"].as<bool>();
  } else {
    training_rule_.sparse_.use_quantized_embedding_ = false;
  }

  // slots
  {
    vector<string> str_slots = absl::StrSplit(conf["plugins"]["slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.base_slots_.resize(str_slots.size());
    for (size_t i = 0; i < str_slots.size(); ++i) {
      absl::SimpleAtoi(str_slots[i], &(training_rule_.sparse_.base_slots_[i]));
    }
  }
  {
    vector<string> str_addition_slots = absl::StrSplit(conf["plugins"]["addition_slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.addition_slots_.resize(str_addition_slots.size());
    for (size_t i = 0; i < str_addition_slots.size(); ++i) {
      absl::SimpleAtoi(str_addition_slots[i], &(training_rule_.sparse_.addition_slots_[i]));
    }
  }
  {
    vector<string> str_memory_slots = absl::StrSplit(conf["plugins"]["memory_slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.memory_slots_.resize(str_memory_slots.size());
    for (size_t i = 0; i < str_memory_slots.size(); ++i) {
      absl::SimpleAtoi(str_memory_slots[i], &(training_rule_.sparse_.memory_slots_[i]));
    }
  }

  // cvm
  training_rule_.sparse_.cvm_rule_.joint_train_ = conf["plugins"]["cvm_plugin"]["joint_train"].as<bool>();
  training_rule_.sparse_.cvm_rule_.decay_rate_  = conf["plugins"]["cvm_plugin"]["decay_rate"].as<float>();

  // lr
  training_rule_.sparse_.lr_rule_.joint_train_   = conf["plugins"]["lr_plugin"]["joint_train"].as<bool>();
  training_rule_.sparse_.lr_rule_.version_aware_ = conf["plugins"]["lr_plugin"]["version_aware"].as<bool>();
  training_rule_.sparse_.lr_rule_.learning_rate_ = conf["plugins"]["lr_plugin"]["learning_rate"].as<float>();
  training_rule_.sparse_.lr_rule_.initial_g2sum_ = conf["plugins"]["lr_plugin"]["initial_g2sum"].as<float>();
  training_rule_.sparse_.lr_rule_.initial_range_ = conf["plugins"]["lr_plugin"]["initial_range"].as<float>();
  if (conf["plugins"]["lr_plugin"]["weight_bounds"].is_null()) {
    training_rule_.sparse_.lr_rule_.weight_upper_bound_ = -std::numeric_limits<float>::infinity();
    training_rule_.sparse_.lr_rule_.weight_upper_bound_ = std::numeric_limits<float>::infinity();
  } else {
    std::vector<float> bounds = conf["plugins"]["lr_plugin"]["weight_bounds"].as<std::vector<float> >();
    PCHECK(bounds.size() == 2);
    training_rule_.sparse_.lr_rule_.weight_lower_bound_ = bounds[0];
    training_rule_.sparse_.lr_rule_.weight_upper_bound_ = bounds[1];
  }

  // fm
  training_rule_.sparse_.fm_rule_.joint_train_   = conf["plugins"]["fm_plugin"]["joint_train"].as<bool>();
  training_rule_.sparse_.fm_rule_.version_aware_ = conf["plugins"]["fm_plugin"]["version_aware"].as<bool>();;
  if (conf["plugins"]["fm_plugin"]["slots"].as<std::string>() == "all") {
    training_rule_.sparse_.fm_rule_.slots_ = training_rule_.sparse_.base_slots_;
  } else {
    vector<string> str_fm_slots = absl::StrSplit(conf["plugins"]["fm_plugin"]["slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.fm_rule_.slots_.resize(str_fm_slots.size());
    for (size_t i = 0; i < str_fm_slots.size(); ++i) {
      absl::SimpleAtoi(str_fm_slots[i], &(training_rule_.sparse_.fm_rule_.slots_[i]));
    }
  }
  training_rule_.sparse_.fm_rule_.dim_ = conf["plugins"]["fm_plugin"]["dim"].as<int>();
  training_rule_.sparse_.fm_rule_.create_threshold_ = conf["plugins"]["fm_plugin"]["create_threshold"].as<float>();
  training_rule_.sparse_.fm_rule_.learning_rate_    = conf["plugins"]["fm_plugin"]["learning_rate"].as<float>();
  training_rule_.sparse_.fm_rule_.initial_g2sum_    = conf["plugins"]["fm_plugin"]["initial_g2sum"].as<float>();
  training_rule_.sparse_.fm_rule_.initial_range_    = conf["plugins"]["fm_plugin"]["initial_range"].as<float>();
  if (conf["plugins"]["fm_plugin"]["weight_bounds"].is_null()) {
    training_rule_.sparse_.fm_rule_.weight_upper_bound_ = -std::numeric_limits<float>::infinity();
    training_rule_.sparse_.fm_rule_.weight_upper_bound_ = std::numeric_limits<float>::infinity();
  } else {
    std::vector<float> bounds = conf["plugins"]["fm_plugin"]["weight_bounds"].as<std::vector<float> >();
    PCHECK(bounds.size() == 2);
    training_rule_.sparse_.fm_rule_.weight_lower_bound_ = bounds[0];
    training_rule_.sparse_.fm_rule_.weight_upper_bound_ = bounds[1];
  }

  // mf
  training_rule_.sparse_.mf_rule_.joint_train_   = conf["plugins"]["mf_plugin"]["joint_train"].as<bool>();
  training_rule_.sparse_.mf_rule_.version_aware_ = conf["plugins"]["mf_plugin"]["version_aware"].as<bool>();;
  if (conf["plugins"]["mf_plugin"]["slots"].as<std::string>() == "all") {
    training_rule_.sparse_.mf_rule_.slots_ = training_rule_.sparse_.base_slots_;
  } else {
    vector<string> str_mf_slots = absl::StrSplit(conf["plugins"]["mf_plugin"]["slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.mf_rule_.slots_.resize(str_mf_slots.size());
    for (size_t i = 0; i < str_mf_slots.size(); ++i) {
      absl::SimpleAtoi(str_mf_slots[i], &(training_rule_.sparse_.mf_rule_.slots_[i]));
    }
  }
  training_rule_.sparse_.mf_rule_.dim_ = conf["plugins"]["mf_plugin"]["dim"].as<int>();
  training_rule_.sparse_.mf_rule_.create_threshold_ = conf["plugins"]["mf_plugin"]["create_threshold"].as<float>();
  training_rule_.sparse_.mf_rule_.learning_rate_    = conf["plugins"]["mf_plugin"]["learning_rate"].as<float>();
  training_rule_.sparse_.mf_rule_.initial_g2sum_    = conf["plugins"]["mf_plugin"]["initial_g2sum"].as<float>();
  training_rule_.sparse_.mf_rule_.initial_range_    = conf["plugins"]["mf_plugin"]["initial_range"].as<float>();
  if (conf["plugins"]["mf_plugin"]["weight_bounds"].is_null()) {
    training_rule_.sparse_.mf_rule_.weight_upper_bound_ = -std::numeric_limits<float>::infinity();
    training_rule_.sparse_.mf_rule_.weight_upper_bound_ = std::numeric_limits<float>::infinity();
  } else {
    std::vector<float> bounds = conf["plugins"]["mf_plugin"]["weight_bounds"].as<std::vector<float> >();
    PCHECK(bounds.size() == 2);
    training_rule_.sparse_.mf_rule_.weight_lower_bound_ = bounds[0];
    training_rule_.sparse_.mf_rule_.weight_upper_bound_ = bounds[1];
  }

  // dic
  training_rule_.sparse_.dic_rule_.version_aware_ = conf["plugins"]["dic_plugin"]["version_aware"].as<bool>();;
  training_rule_.sparse_.dic_rule_.dim_ = conf["plugins"]["dic_plugin"]["dim"].as<int>();
  training_rule_.sparse_.dic_rule_.delete_after_silent_days_ = conf["plugins"]["dic_plugin"]["delete_after_silent_days"].as<int>();
  training_rule_.sparse_.dic_rule_.create_threshold_ = conf["plugins"]["dic_plugin"]["create_threshold"].as<float>();
  training_rule_.sparse_.dic_rule_.delete_threshold_ = conf["plugins"]["dic_plugin"]["delete_threshold"].as<float>();
  training_rule_.sparse_.dic_rule_.join_threshold_   = conf["plugins"]["dic_plugin"]["join_threshold"].as<float>();
  training_rule_.sparse_.dic_rule_.decay_rate_       = conf["plugins"]["dic_plugin"]["decay_rate"].as<float>();
  training_rule_.sparse_.dic_rule_.learning_rate_    = conf["plugins"]["dic_plugin"]["learning_rate"].as<float>();
  training_rule_.sparse_.dic_rule_.initial_range_    = conf["plugins"]["dic_plugin"]["initial_range"].as<float>();
  training_rule_.sparse_.dic_rule_.mom_decay_rate_   = conf["plugins"]["dic_plugin"]["mom_decay_rate"].as<float>();
  training_rule_.sparse_.dic_rule_.ada_decay_rate_   = conf["plugins"]["dic_plugin"]["ada_decay_rate"].as<float>();
  training_rule_.sparse_.dic_rule_.ada_epsilon_      = conf["plugins"]["dic_plugin"]["ada_epsilon"].as<float>();
  training_rule_.sparse_.dic_rule_.initial_g2sum_    = conf["plugins"]["dic_plugin"]["initial_g2sum"].as<float>();
  if (conf["plugins"]["dic_plugin"]["weight_bounds"].is_null()) {
    training_rule_.sparse_.dic_rule_.weight_upper_bound_ = -std::numeric_limits<float>::infinity();
    training_rule_.sparse_.dic_rule_.weight_upper_bound_ = std::numeric_limits<float>::infinity();
  } else {
    std::vector<float> bounds = conf["plugins"]["dic_plugin"]["weight_bounds"].as<std::vector<float> >();
    PCHECK(bounds.size() == 2);
    training_rule_.sparse_.dic_rule_.weight_lower_bound_ = bounds[0];
    training_rule_.sparse_.dic_rule_.weight_upper_bound_ = bounds[1];
  }

  // wide
  training_rule_.sparse_.wide_rule_.joint_train_   = conf["plugins"]["wide_plugin"]["joint_train"].as<bool>();
  training_rule_.sparse_.wide_rule_.version_aware_ = conf["plugins"]["wide_plugin"]["version_aware"].as<bool>();;
  if (conf["plugins"]["wide_plugin"]["slots"].as<std::string>() == "all") {
    training_rule_.sparse_.wide_rule_.slots_ = training_rule_.sparse_.base_slots_;
  } else {
    vector<string> str_wide_slots = absl::StrSplit(conf["plugins"]["wide_plugin"]["slots"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.sparse_.wide_rule_.slots_.resize(str_wide_slots.size());
    for (size_t i = 0; i < str_wide_slots.size(); ++i) {
      absl::SimpleAtoi(str_wide_slots[i], &(training_rule_.sparse_.wide_rule_.slots_[i]));
    }
  }
  training_rule_.sparse_.wide_rule_.learning_rate_ = conf["plugins"]["wide_plugin"]["learning_rate"].as<float>();
  training_rule_.sparse_.wide_rule_.initial_g2sum_ = conf["plugins"]["wide_plugin"]["initial_g2sum"].as<float>();
  training_rule_.sparse_.wide_rule_.initial_range_ = conf["plugins"]["wide_plugin"]["initial_range"].as<float>();
  if (conf["plugins"]["wide_plugin"]["weight_bounds"].is_null()) {
    training_rule_.sparse_.wide_rule_.weight_upper_bound_ = -std::numeric_limits<float>::infinity();
    training_rule_.sparse_.wide_rule_.weight_upper_bound_ = std::numeric_limits<float>::infinity();
  } else {
    std::vector<float> bounds = conf["plugins"]["wide_plugin"]["weight_bounds"].as<std::vector<float> >();
    PCHECK(bounds.size() == 2);
    training_rule_.sparse_.wide_rule_.weight_lower_bound_ = bounds[0];
    training_rule_.sparse_.wide_rule_.weight_upper_bound_ = bounds[1];
  }

  // dnn
  if (conf["plugins"]["dnn_plugin"].is_scalar()) {
    *conf["plugins"]["dnn_plugin"] = YAML::LoadFile(conf["plugins"]["dnn_plugin"].as<std::string>());
  }

  training_rule_.dense_.q_names_ = absl::StrSplit(conf["plugins"]["dnn_plugin"]["q_names"].as<std::string>(), ' ', absl::SkipWhitespace());
  if (conf["plugins"]["dnn_plugin"]["q_weight"].is_defined()) {
    vector<string> str_q_weight = absl::StrSplit(conf["plugins"]["dnn_plugin"]["q_weight"].as<string>(), ' ', absl::SkipWhitespace());
    training_rule_.dense_.q_weight_.resize(str_q_weight.size());
    for (size_t i = 0; i < str_q_weight.size(); ++i) {
      absl::SimpleAtof(str_q_weight[i], &(training_rule_.dense_.q_weight_[i]));
    }
  } else {
    training_rule_.dense_.q_weight_ = std::vector<float>(training_rule_.dense_.q_names_.size(), 1.0);
  }
  PCHECK(training_rule_.dense_.q_weight_.size() == training_rule_.dense_.q_names_.size());

  if(conf["plugins"]["dnn_plugin"]["gaussian_outputs"].is_defined()) {
    training_rule_.dense_.use_gaussian_ = true;
    training_rule_.dense_.gaussian_output_names_ = absl::StrSplit(conf["plugins"]["dnn_plugin"]["gaussian_outputs"].as<std::string>(), ' ', absl::SkipWhitespace());
  } else {
    training_rule_.dense_.use_gaussian_   = false;
  }
  if(conf["plugins"]["dnn_plugin"]["memory_outputs"].is_defined()) {
    training_rule_.dense_.memory_output_names_ = absl::StrSplit(conf["plugins"]["dnn_plugin"]["memory_outputs"].as<std::string>(), ' ', absl::SkipWhitespace());
  }
  if(conf["plugins"]["dnn_plugin"]["use_wide"].is_defined()) {
    training_rule_.dense_.use_wide_ = conf["plugins"]["dnn_plugin"]["use_wide"].as<bool>();
  } else {
    training_rule_.dense_.use_wide_ = false;
  }

  training_rule_.dense_.back_propagate_input_ = conf["plugins"]["dnn_plugin"]["back_propagate_input"].as<bool>();
  training_rule_.dense_.back_addition_input_  = conf["plugins"]["dnn_plugin"]["back_addition_input"].as<bool>();
  training_rule_.dense_.back_memory_input_    = conf["plugins"]["dnn_plugin"]["back_memory_input"].as<bool>();

  training_rule_.dense_.global_init_range_  = conf["plugins"]["dnn_plugin"]["global_init_range"].as<float>();

  training_rule_.dense_.learning_rate_  = conf["plugins"]["dnn_plugin"]["learning_rate"].as<float>();
  training_rule_.dense_.avg_decay_rate_ = conf["plugins"]["dnn_plugin"]["avg_decay_rate"].as<float>();
  training_rule_.dense_.ada_decay_rate_ = conf["plugins"]["dnn_plugin"]["ada_decay_rate"].as<float>();
  training_rule_.dense_.ada_epsilon_    = conf["plugins"]["dnn_plugin"]["ada_epsilon"].as<float>();
  training_rule_.dense_.mom_decay_rate_ = conf["plugins"]["dnn_plugin"]["mom_decay_rate"].as<float>();
  if (conf["plugins"]["dnn_plugin"]["weight_decay"].is_defined()) {
    training_rule_.dense_.weight_decay_ = conf["plugins"]["dnn_plugin"]["weight_decay"].as<float>();
  } else {
    training_rule_.dense_.weight_decay_ = 0.0;
  }
  if (conf["plugins"]["dnn_plugin"]["optimizer"].is_defined()) {
    training_rule_.dense_.optimizer_ = conf["plugins"]["dnn_plugin"]["optimizer"].as<std::string>();
  } else {
    training_rule_.dense_.optimizer_ = "";
  }

  training_rule_.dense_.summary_decay_rate_          = conf["plugins"]["dnn_plugin"]["summary_decay_rate"].as<float>();
  training_rule_.dense_.summary_squared_sum_epsilon_ = conf["plugins"]["dnn_plugin"]["summary_squared_sum_epsilon"].as<float>();
  training_rule_.dense_.summary_init_n_              = conf["plugins"]["dnn_plugin"]["summary_init_n"].as<float>();
  training_rule_.dense_.summary_init_squared_sum_    = conf["plugins"]["dnn_plugin"]["summary_init_squared_sum"].as<float>();

  training_rule_.dense_.test_layers_at_joining_   = absl::StrSplit(conf["plugins"]["dnn_plugin"]["test_layers_at_joining"].as<std::string>(), ' ');
  training_rule_.dense_.train_layers_at_joining_  = absl::StrSplit(conf["plugins"]["dnn_plugin"]["train_layers_at_joining"].as<std::string>(), ' ');
  training_rule_.dense_.test_layers_at_updating_  = absl::StrSplit(conf["plugins"]["dnn_plugin"]["test_layers_at_updating"].as<std::string>(), ' ');
  training_rule_.dense_.train_layers_at_updating_ = absl::StrSplit(conf["plugins"]["dnn_plugin"]["train_layers_at_updating"].as<std::string>(), ' ');

  if (conf["plugins"]["dnn_plugin"]["vec_input"].is_defined()) {
    int vec_input_size = conf["plugins"]["dnn_plugin"]["vec_input"].size();
    for (int i = 0; i < vec_input_size; ++i) {
      std::string name = conf["plugins"]["dnn_plugin"]["vec_input"][i]["name"].as<std::string>();
      int dim          = conf["plugins"]["dnn_plugin"]["vec_input"][i]["dim"].as<int>();
      training_rule_.dense_.vec_input_.push_back({name, dim});
    }
  }

  if (conf["plugins"]["dnn_plugin"]["addition_input"].is_defined()) {
    int addition_input_size = conf["plugins"]["dnn_plugin"]["addition_input"].size();
    for (int i = 0; i < addition_input_size; ++i) {
      std::string name = conf["plugins"]["dnn_plugin"]["addition_input"][i]["name"].as<std::string>();
      int dim          = conf["plugins"]["dnn_plugin"]["addition_input"][i]["dim"].as<int>();
      int vlen         = conf["plugins"]["dnn_plugin"]["addition_input"][i]["vlen"].as<int>();
      training_rule_.dense_.addition_input_.push_back({name, dim, vlen});
    }
  }

  int summary_size = conf["plugins"]["dnn_plugin"]["summary"].size();
  for (int i = 0; i < summary_size; ++i) {
    string name = conf["plugins"]["dnn_plugin"]["summary"][i]["name"].as<string>();
    int length  = conf["plugins"]["dnn_plugin"]["summary"][i]["len"].as<int>();
    training_rule_.dense_.summary_.push_back({name, length});
  }

  int param_size = conf["plugins"]["dnn_plugin"]["param"].size();
  for (int i = 0; i < param_size; ++i) {
    string name         = conf["plugins"]["dnn_plugin"]["param"][i]["name"].as<string>();
    int row_n           = conf["plugins"]["dnn_plugin"]["param"][i]["rown"].as<int>();
    int col_n           = conf["plugins"]["dnn_plugin"]["param"][i]["coln"].as<int>();
    bool scale_by_row_n = conf["plugins"]["dnn_plugin"]["param"][i]["scale_by_rown"].as<bool>();
    float init_range    = conf["plugins"]["dnn_plugin"]["param"][i]["init_range"].as<float>();
    training_rule_.dense_.param_.push_back({name, row_n, col_n, init_range, scale_by_row_n});
  }

  training_rule_.dense_.all_layers_conf_ = conf["plugins"]["dnn_plugin"]["layer"];
  int layer_size = conf["plugins"]["dnn_plugin"]["layer"].size();
  for (int i = 0; i < layer_size; ++i) {
    training_rule_.dense_.layer_conf_.push_back(conf["plugins"]["dnn_plugin"]["layer"][i]);
  }
}

void ConfigManager::load_worker_conf(Config& conf) {
  worker_rule_.batch_size_               = conf["minibatch_size"].as<int>();
  worker_rule_.drop_feature_             = (bool)(conf["drop_feature"].as<int>());
  worker_rule_.load_balance_             = conf["load_balance"].as<bool>();
  worker_rule_.bias_slot_                = conf["bias_slot"].as<int>();
  worker_rule_.position_slot_            = conf["position_slot"].as<int>();

  vector<string> str_position_feas = absl::StrSplit(conf["position_feas"].as<string>(), ' ', absl::SkipWhitespace());
  worker_rule_.position_feas_.resize(str_position_feas.size());
  for (size_t i = 0; i < str_position_feas.size(); ++i) {
    absl::SimpleAtoi(str_position_feas[i], &(worker_rule_.position_feas_[i]));
  }

  if (conf["data_shuffler"].is_defined()) {
    worker_rule_.data_shuffler_rule_.merge_by_lineid_               = conf["data_shuffler"]["merge_by_lineid"].as<bool>();
    worker_rule_.data_shuffler_rule_.check_show_clk_consistency_    = conf["data_shuffler"]["check_show_clk_consistency"].as<bool>();
    worker_rule_.data_shuffler_rule_.erase_duplicate_feas_          = conf["data_shuffler"]["erase_duplicate_feas"].as<bool>();
    worker_rule_.data_shuffler_rule_.delete_instance_with_out_slot_ = conf["data_shuffler"]["delete_instances_without_slot"].as<string>();
  }

  worker_rule_.train_mode_ = conf["train_mode"].as<string>();
  if (conf["offline_runner"].is_defined()) {
    worker_rule_.offline_worker_rule_.shuffle_data_      = conf["offline_runner"]["shuffle_data"].as<bool>();
    worker_rule_.offline_worker_rule_.test_per_step_     = conf["offline_runner"]["test_per_step"].as<bool>();
    worker_rule_.offline_worker_rule_.load_prior_model_  = conf["offline_runner"]["load_prior_model"].as<bool>();
    worker_rule_.offline_worker_rule_.recover_mode_      = conf["offline_runner"]["recover_mode"].as<bool>();
    worker_rule_.offline_worker_rule_.model_path_        = conf["offline_runner"]["model_path"].as<string>();
    worker_rule_.offline_worker_rule_.model_donefile_    = conf["offline_runner"]["model_donefile"].as<string>();
    worker_rule_.offline_worker_rule_.model_converter_   = conf["offline_runner"]["load_model_converter"].as<string>();
    worker_rule_.offline_worker_rule_.data_path_         = conf["offline_runner"]["data_path"].as<string>();
    worker_rule_.offline_worker_rule_.data_donefile_     = conf["offline_runner"]["data_done_file"].as<string>();
    worker_rule_.offline_worker_rule_.data_converter_    = conf["offline_runner"]["data_converter"].as<string>();                  ;

    worker_rule_.offline_worker_rule_.start_update_days_ = conf["offline_runner"]["start_update_days"].as<int>();
    worker_rule_.offline_worker_rule_.train_days_        = conf["offline_runner"]["train_days"].as<string>();
    worker_rule_.offline_worker_rule_.test_days_         = conf["offline_runner"]["test_days"].as<string>();
    worker_rule_.offline_worker_rule_.model_days_        = conf["offline_runner"]["model_days"].as<string>();

    if (conf["offline_runner"]["dump_result"].is_defined()) {
      worker_rule_.offline_worker_rule_.dump_result_      = conf["offline_runner"]["dump_result"].as<bool>();
      worker_rule_.offline_worker_rule_.dump_result_path_ = conf["offline_runner"]["result_data_path"].as<string>();
    } else {
      worker_rule_.offline_worker_rule_.dump_result_      = false;
      worker_rule_.offline_worker_rule_.dump_result_path_ = "";
    }

    if (conf["offline_runner"]["open_sample"].is_defined()) {
      worker_rule_.offline_worker_rule_.open_sampling_  = conf["offline_runner"]["open_sample"].as<bool>();
      worker_rule_.offline_worker_rule_.sampling_rate_  = conf["offline_runner"]["sample_rate"].as<float>();
    } else {
      worker_rule_.offline_worker_rule_.open_sampling_  = false;
      worker_rule_.offline_worker_rule_.sampling_rate_  = 1.0;
    }

    worker_rule_.offline_worker_rule_.join_data_path_   = conf["offline_runner"]["join_data_path"].as<string>();
    worker_rule_.offline_worker_rule_.backdate_         = conf["offline_runner"]["backdate"].as<bool>();
    worker_rule_.offline_worker_rule_.backdate_days_    = conf["offline_runner"]["backdate_days"].as<string>();
  }
  if (conf["online_runner"].is_defined()) {
    worker_rule_.online_worker_rule_.shuffle_data_      = conf["online_runner"]["shuffle_data"].as<bool>();
    worker_rule_.online_worker_rule_.test_per_step_     = conf["online_runner"]["test_per_step"].as<bool>();
    worker_rule_.online_worker_rule_.load_prior_model_  = conf["online_runner"]["load_prior_model"].as<bool>();
    worker_rule_.online_worker_rule_.recover_mode_      = conf["online_runner"]["recover_model"].as<bool>();
    worker_rule_.online_worker_rule_.model_path_        = conf["online_runner"]["model_path"].as<string>();
    worker_rule_.online_worker_rule_.model_donefile_    = conf["online_runner"]["model_donefile"].as<string>();
    worker_rule_.online_worker_rule_.model_converter_   = conf["online_runner"]["load_model_converter"].as<string>();
    worker_rule_.online_worker_rule_.data_path_         = conf["online_runner"]["data_path"].as<string>();
    worker_rule_.online_worker_rule_.data_donefile_     = conf["online_runner"]["data_done_file"].as<string>();
    worker_rule_.online_worker_rule_.data_converter_    = conf["online_runner"]["data_converter"].as<string>();                  ;

    worker_rule_.online_worker_rule_.start_update_days_ = conf["online_runner"]["start_update_days"].as<int>();
    worker_rule_.online_worker_rule_.train_days_        = conf["online_runner"]["train_days"].as<string>();
    worker_rule_.online_worker_rule_.test_days_         = conf["online_runner"]["test_days"].as<string>();
    worker_rule_.online_worker_rule_.model_days_        = conf["online_runner"]["model_days"].as<string>();

    if (conf["online_runner"]["dump_result"].is_defined()) {
      worker_rule_.online_worker_rule_.dump_result_       = conf["online_runner"]["dump_result"].as<bool>();
      worker_rule_.online_worker_rule_.dump_result_path_  = conf["online_runner"]["result_data_path"].as<string>();
    } else {
      worker_rule_.online_worker_rule_.dump_result_       = false;
      worker_rule_.online_worker_rule_.dump_result_path_  = "";
    }

    if (conf["online_runner"]["open_sample"].is_defined()) {
      worker_rule_.online_worker_rule_.open_sampling_  = conf["online_runner"]["open_sample"].as<bool>();
      worker_rule_.online_worker_rule_.sampling_rate_  = conf["online_runner"]["sample_rate"].as<float>();
    } else {
      worker_rule_.online_worker_rule_.open_sampling_  = false;
      worker_rule_.online_worker_rule_.sampling_rate_  = 1.0;
    }

    worker_rule_.online_worker_rule_.join_data_path_    = conf["online_runner"]["join_data_path"].as<string>();
    worker_rule_.online_worker_rule_.backdate_          = conf["online_runner"]["backdate"].as<bool>();
    worker_rule_.online_worker_rule_.backdate_days_     = conf["online_runner"]["backdate_days"].as<string>();

    worker_rule_.online_worker_rule_.q_diff_left_threshold_  = conf["online_runner"]["q_diff_left_threshold"].as<float>();
    worker_rule_.online_worker_rule_.q_diff_right_threshold_ = conf["online_runner"]["q_diff_right_threshold"].as<float>();
    worker_rule_.online_worker_rule_.auc_diff_threshold_     = conf["online_runner"]["auc_diff_threshold"].as<float>();
    worker_rule_.online_worker_rule_.q_check_index_          = conf["online_runner"]["q_check_index"].as<int>();

    worker_rule_.online_worker_rule_.mq_reader_path_         = conf["online_runner"]["mq_reader_path"].as<string>();
    worker_rule_.online_worker_rule_.data_block_size_        = conf["online_runner"]["data_block_size"].as<int64_t>();
    worker_rule_.online_worker_rule_.test_step_interval_     = conf["online_runner"]["test_step_interval"].as<int>();
    worker_rule_.online_worker_rule_.dump_model_interval_    = conf["online_runner"]["dump_model_interval"].as<int>();
    worker_rule_.online_worker_rule_.fetch_data_timeout_     = conf["online_runner"]["fetch_data_timeout"].as<int64_t>();
    worker_rule_.online_worker_rule_.mq_reader_name_         = conf["online_runner"]["mq_reader_name"].as<string>();

    worker_rule_.online_worker_rule_.batch_model_timestamp_  = conf["online_runner"]["batch_model_timestamp"].as<string>();
    worker_rule_.online_worker_rule_.start_dump_timestamp_   = conf["online_runner"]["start_dump_timestamp"].as<string>();
    worker_rule_.online_worker_rule_.dump_timestamp_         = conf["online_runner"]["dump_timestamp"].as<string>();
    worker_rule_.online_worker_rule_.dump_time_interval_     = conf["online_runner"]["dump_time_interval"].as<int64_t>();
    worker_rule_.online_worker_rule_.dump_time_diff_         = conf["online_runner"]["dump_time_diff"].as<int64_t>();
    worker_rule_.online_worker_rule_.read_time_shift_        = conf["online_runner"]["read_time_shift"].as<int64_t>();

    worker_rule_.online_worker_rule_.max_null_fetch_iteration_ = conf["online_runner"]["max_null_fetch_interation"].as<int>();
    worker_rule_.online_worker_rule_.mq_queue_name_            = conf["online_runner"]["mq_queue_name"].as<string>();
    worker_rule_.online_worker_rule_.mq_customize_options_     = conf["online_runner"]["mq_customize_options"].as<string>();
    worker_rule_.online_worker_rule_.user_identifier_          = conf["online_runner"]["user_identifier"].as<string>();
    worker_rule_.online_worker_rule_.log_print_interval_       = conf["online_runner"]["log_print_interval"].as<int>();
  }
}

void ConfigManager::initialize(const std::string& config_file, bool is_worker) {
  if (is_inited_) {
    return;
  }
  is_inited_ = true;

  // param_server or worker
  regist_is_worker(is_worker);

  Config conf;
  *conf = YAML::LoadFile(config_file);

  load_framework_conf(conf);
  load_plugins_conf(conf);

  if (pick_is_worker()) {
    load_worker_conf(conf);
  }
}

void ConfigManager::finalize() {
  is_inited_ = false;
  is_worker_ = false;

  rpc_server_info_.clear();

  {
    struct RuntimeConfig tmp;
    runtime_config_ = tmp;
  }

  {
    struct ResourceConfig tmp;
    resource_config_ = tmp;
  }

  {
    struct TrainingRule tmp;
    training_rule_ = tmp;
  }

  {
    struct WorkerRule tmp;
    worker_rule_ = tmp;
  }
}

// is inited
void ConfigManager::regist_is_inited(const bool is_inited) {
  is_inited_ = is_inited;
}
bool ConfigManager::pick_is_inited() {
  return is_inited_;
}

// param server or worker
void ConfigManager::regist_is_worker(const bool is_worker) {
  is_worker_ = is_worker;
}
bool ConfigManager::pick_is_worker() {
  return is_worker_;
}

void ConfigManager::regist_rpc_server_info(const vector<RPCServerInfo>& rpc_server_info) {
  rpc_server_info_ = rpc_server_info;
}
const vector<RPCServerInfo>& ConfigManager::pick_rpc_server_info() {
  return rpc_server_info_;
}

// runtime config
void ConfigManager::regist_local_thread_num(const int local_thread_num) {
  runtime_config_.local_thread_num_ = local_thread_num;
}
void ConfigManager::regist_write_thread_num(const int write_thread_num) {
  runtime_config_.write_thread_num_ = write_thread_num;
}
void ConfigManager::regist_disk_buffer_size(const size_t disk_buffer_size) {
  runtime_config_.disk_buffer_size_ = disk_buffer_size;
}
void ConfigManager::regist_hdfs_buffer_size(const size_t hdfs_buffer_size) {
  runtime_config_.hdfs_buffer_size_ = hdfs_buffer_size;
}
void ConfigManager::regist_hdfs_command(const string& hdfs_command) {
  runtime_config_.hdfs_command_ = hdfs_command;
}
void ConfigManager::regist_hdfs_reader_command(const string& hdfs_reader_command) {
  runtime_config_.hdfs_reader_command_ = hdfs_reader_command;
}
int ConfigManager::pick_local_thread_num() {
  return runtime_config_.local_thread_num_;
}
int ConfigManager::pick_write_thread_num() {
  return runtime_config_.write_thread_num_;
}
size_t ConfigManager::pick_disk_buffer_size() {
  return runtime_config_.disk_buffer_size_;
}
size_t ConfigManager::pick_hdfs_buffer_size() {
  return runtime_config_.hdfs_buffer_size_;
}
const string& ConfigManager::pick_hdfs_command() {
  return runtime_config_.hdfs_command_;
}
const string& ConfigManager::pick_hdfs_reader_command() {
  return runtime_config_.hdfs_reader_command_;
}

void ConfigManager::regist_data_reader_default_capacity(size_t capacity) {
  runtime_config_.data_reader_default_capacity_ = capacity;
}
void ConfigManager::regist_data_reader_default_block_size(int block_size) {
  runtime_config_.data_reader_default_block_size_ = block_size;
}
void ConfigManager::regist_data_reader_default_thread_num(int thread_num) {
  runtime_config_.data_reader_default_thread_num_ = thread_num;
}
size_t ConfigManager::pick_data_reader_default_capacity() {
  return runtime_config_.data_reader_default_capacity_;
}
int ConfigManager::pick_data_reader_default_block_size() {
  return runtime_config_.data_reader_default_block_size_;
}
int ConfigManager::pick_data_reader_default_thread_num() {
  return runtime_config_.data_reader_default_thread_num_;
}

// resource config
void ConfigManager::regist_global_shard_num(const int global_shard_num) {
  resource_config_.global_shard_num_ = global_shard_num;
}
void ConfigManager::regist_local_shard_num(const int local_shard_num) {
  resource_config_.local_shard_num_ = local_shard_num;
}

void ConfigManager::regist_shard_info(const int mpi_size, const int mpi_rank) {
  regist_global_shard_num(resource_config_.local_shard_num_ * mpi_size);

  resource_config_.local_shard_info_.resize(resource_config_.local_shard_num_);
  resource_config_.global_shard_info_.resize(resource_config_.global_shard_num_);

  for (int i = 0; i < resource_config_.local_shard_num_; ++i) {
    int global_id = i * mpi_size + mpi_rank;
    resource_config_.local_shard_info_[i]  = (struct ShardInfo) { global_id, i, mpi_rank };
    resource_config_.global_shard_info_[global_id] = (struct ShardInfo) { global_id, i, mpi_rank };
  }
}

const int ConfigManager::pick_global_shard_num() {
  return resource_config_.global_shard_num_;
}
const int ConfigManager::pick_local_shard_num() {
  return resource_config_.local_shard_num_;
}
const ShardInfo& ConfigManager::pick_local_shard_info(const int local_shard_id) {
  CHECK(local_shard_id >= 0 && local_shard_id < resource_config_.local_shard_num_);
  return resource_config_.local_shard_info_[local_shard_id];
}
const ShardInfo& ConfigManager::pick_global_shard_info(const int global_shard_id) {
  CHECK(global_shard_id >= 0 && global_shard_id < resource_config_.global_shard_num_);
  return resource_config_.global_shard_info_[global_shard_id];
}


// param server config
void ConfigManager::regist_training_rule(const TrainingRule& rule) {
  training_rule_ = rule;
}
const TrainingRule& ConfigManager::pick_training_rule() {
  return training_rule_;
}

// worker config
void ConfigManager::regist_worker_rule(const WorkerRule& rule) {
  worker_rule_ = rule;
}
const WorkerRule& ConfigManager::pick_worker_rule() {
  return worker_rule_;
}

} // namespace runtime
} // namespace ps

