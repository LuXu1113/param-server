#include "model/distributed_learner/rtsparse_learner.h"

#include "absl/time/time.h"
#include "absl/strings/numbers.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/rpc_agent.h"
#include "toolkit/fs_agent.h"
#include "toolkit/thread_group.h"
#include "runtime/config_manager.h"
#include "model/data/component.h"

using std::vector;
using std::string;
using std::unordered_map;
using std::max;
using ps::toolkit::Channel;
using ps::toolkit::MPIAgent;
using ps::toolkit::RPCAgent;
using ps::toolkit::FSAgent;
using ps::toolkit::parallel_run;
using ps::runtime::ConfigManager;
using ps::runtime::TrainingRule;
using ps::param_table::SparseFeatureVer1;
using ps::param_table::SparseValueVer1;
using ps::param_table::SparseEmbeddingVer1;
using ps::param_table::DenseValueVer1;
using ps::param_table::SummaryValueVer1;

namespace ps {
namespace model {

void RTSparseLearner::initialize() {
  CHECK(!is_initialized_);
  MPIAgent::mpi_barrier_group();

  RPCAgent::initialize(ConfigManager::pick_rpc_server_info());
  MPIAgent::mpi_barrier_group();

  const TrainingRule& rule = ConfigManager::pick_training_rule();
  for (size_t i = 0; i < rule.sparse_.base_slots_.size(); ++i) {
    base_slot_set_.set(rule.sparse_.base_slots_[i], i);
  }
  for (size_t i = 0; i < rule.sparse_.memory_slots_.size(); ++i) {
    memory_slot_set_.set(rule.sparse_.memory_slots_[i], i);
  }

  cvm_plugin_.initialize();
  lr_plugin_.initialize();
  fm_plugin_.initialize();
  mf_plugin_.initialize();
  wide_plugin_.initialize();
  ps_dnn_plugin_.initialize();
  MPIAgent::mpi_barrier_group();

  initialize_param_table();
  initialize_thread_local_data();

  lr_auc_.set_name("LR");
  mf_auc_.set_name("MF");
  fm_auc_.set_name("FM");
  adq_auc_.set_name("ADQ");
  dnn_auc_num_ = ConfigManager::pick_training_rule().dense_.q_names_.size();
  dnn_auc_ = new AUCCalculator[dnn_auc_num_];
  for (int i = 0; i < dnn_auc_num_; ++i) {
    dnn_auc_[i].set_name(ConfigManager::pick_training_rule().dense_.q_names_[i]);
  }

  perf_preprocess_.set_name("preprocess");
  perf_pull_.set_name("pull");
  perf_push_.set_name("push");
  perf_forward_.set_name("forward");
  perf_back_propagate_.set_name("back_propagate");
  perf_auc_.set_name("auc");
  perf_pull_dense_.set_name("pull dense");
  perf_push_dense_.set_name("push dense");
  perf_pull_sparse_.set_name("pull sparse");
  perf_push_sparse_.set_name("push sparse");

  use_sync_comm_ = false;
  is_initialized_ = true;

  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::finalize() {
  CHECK(is_initialized_);
  MPIAgent::mpi_barrier_group();

  cvm_plugin_.finalize();
  lr_plugin_.finalize();
  fm_plugin_.finalize();
  mf_plugin_.finalize();
  wide_plugin_.finalize();
  ps_dnn_plugin_.finalize();
  MPIAgent::mpi_barrier_group();

  finalize_thread_local_data();
  MPIAgent::mpi_barrier_group();

  finalize_param_table();
  MPIAgent::mpi_barrier_group();

  delete(dnn_auc_);
  MPIAgent::mpi_barrier_group();

  LOG(INFO) << "shutdown param server ...";
  if (MPIAgent::mpi_rank_group() == 0) {
    RPCAgent::shutdown();
  }
  MPIAgent::mpi_barrier_group();
  LOG(INFO) << "shutdown param server finished ...";

  RPCAgent::finalize();
  MPIAgent::mpi_barrier_group();

  is_initialized_ = false;
}

void RTSparseLearner::begin_day() {
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::end_day() {
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::time_decay() {
  sparse_table_client_.time_decay();
  memory_table_client_.time_decay();
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::shrink_table() {
  uint64_t sparse_feature_num_before_shrink = 0;
  uint64_t sparse_feature_num_after_shrink  = 0;
  uint64_t memory_feature_num_before_shrink = 0;
  uint64_t memory_feature_num_after_shrink  = 0;

  if (MPIAgent::mpi_rank_group() == 0) {
    sparse_feature_num_before_shrink = sparse_table_client_.feature_num();
    memory_feature_num_before_shrink = memory_table_client_.feature_num();
  }
  MPIAgent::mpi_barrier_group();

  sparse_table_client_.shrink();
  memory_table_client_.shrink();
  MPIAgent::mpi_barrier_group();

  if (MPIAgent::mpi_rank_group() == 0) {
    sparse_feature_num_after_shrink = sparse_table_client_.feature_num();
    memory_feature_num_after_shrink = memory_table_client_.feature_num();

    fprintf(stdout, "Sparse feature num: %llu -> %llu\n",
            (unsigned long long)sparse_feature_num_before_shrink,
            (unsigned long long)sparse_feature_num_after_shrink);
    fprintf(stdout, "Memory feature num: %llu -> %llu\n",
            (unsigned long long)memory_feature_num_before_shrink,
            (unsigned long long)memory_feature_num_after_shrink);
  }
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::begin_pass(TrainingPhase phase) {
  phase_ = phase;

  // initialize auc calculator
  MPIAgent::mpi_barrier_group();
  lr_auc_.clear();
  mf_auc_.clear();
  fm_auc_.clear();
  adq_auc_.clear();
  for (int i = 0; i < dnn_auc_num_; ++i) {
    dnn_auc_[i].clear();
  }
  MPIAgent::mpi_barrier_group();

  // initialize operating log
  perf_preprocess_.clear();
  perf_pull_.clear();
  perf_push_.clear();
  perf_forward_.clear();
  perf_back_propagate_.clear();
  perf_auc_.clear();
  perf_pull_dense_.clear();
  perf_pull_sparse_.clear();
  perf_push_dense_.clear();
  perf_push_sparse_.clear();
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::end_pass() {
  // calculate auc
  MPIAgent::mpi_barrier_group();
  lr_auc_.compute();
  mf_auc_.compute();
  fm_auc_.compute();
  adq_auc_.compute();
  if (phase_ == TrainingPhase::JOINING) {
    for (int i = 0; i < dnn_auc_num_; ++i) {
      dnn_auc_[i].compute();
    }
  }
  MPIAgent::mpi_barrier_group();

  // print auc
  if (MPIAgent::mpi_rank_group() == 0) {
    lr_auc_.print_all_measures();
    mf_auc_.print_all_measures();
    fm_auc_.print_all_measures();
    adq_auc_.print_all_measures();
    if (phase_ == TrainingPhase::JOINING) {
      for (int i = 0; i < dnn_auc_num_; ++i) {
        dnn_auc_[i].print_all_measures();
      }
    }
  }
  MPIAgent::mpi_barrier_group();

  // print perf
  perf_preprocess_.log();
  perf_pull_.log();
  perf_push_.log();
  perf_forward_.log();
  perf_back_propagate_.log();
  perf_auc_.log();
  perf_pull_dense_.log();
  perf_pull_sparse_.log();
  perf_push_dense_.log();
  perf_push_sparse_.log();
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::set_testmode(bool mode) {
  test_mode_ = mode;
}

void RTSparseLearner::preprocess(ThreadLocalData *data) {
  cvm_plugin_.preprocess(data);
  lr_plugin_.preprocess(data);
  fm_plugin_.preprocess(data);
  mf_plugin_.preprocess(data);
  wide_plugin_.preprocess(data);
  if (phase_ == TrainingPhase::JOINING) {
    ps_dnn_plugin_.preprocess(data);
  }
}

void RTSparseLearner::feed_forward(ThreadLocalData *data) {
  cvm_plugin_.feed_forward(data);
  lr_plugin_.feed_forward(data);
  fm_plugin_.feed_forward(data);
  mf_plugin_.feed_forward(data);
  wide_plugin_.feed_forward(data);
  if (phase_ == TrainingPhase::JOINING) {
    ps_dnn_plugin_.feed_forward(data);
  }
}

void RTSparseLearner::back_propagate(ThreadLocalData *data) {
  if (test_mode_) {
    return;
  }

  init_pushs(data);

  if (phase_ == TrainingPhase::JOINING) {
    ps_dnn_plugin_.back_propagate(data);
  }
  wide_plugin_.back_propagate(data);
  mf_plugin_.back_propagate(data);
  fm_plugin_.back_propagate(data);
  lr_plugin_.back_propagate(data);
  cvm_plugin_.back_propagate(data);
}

void RTSparseLearner::pull_params(ThreadLocalData *data) {
  absl::Time ts1;
  absl::Time ts2;

  if (phase_ == TrainingPhase::JOINING) {
    ts1 = absl::Now();
    dense_table_client_.pull(&(data->dnn_pulls_));
    summary_table_client_.pull(&(data->dnn_summary_pulls_));
    ts2 = absl::Now();
    perf_pull_dense_.record(ts1, ts2);
  }

  ts1 = absl::Now();
  vector<SparseFeatureVer1> feas;
  vector<SparseValueVer1> fea_pulls;
  vector<SparseFeatureVer1> memory_feas;
  vector<SparseEmbeddingVer1> memory_fea_pulls;

  for (int i = 0; i < data->batch_size_; ++i) {
    feas.insert(feas.end(), data->minibatch_[i].feas_.begin(), data->minibatch_[i].feas_.end());
    memory_feas.insert(memory_feas.end(), data->minibatch_[i].memory_feas_.begin(), data->minibatch_[i].memory_feas_.end());
  }
  sparse_table_client_.pull(feas, &(fea_pulls), (!test_mode_));
  memory_table_client_.pull(memory_feas, &(memory_fea_pulls), (!test_mode_));

  for (int i = 0, j1 = 0, j2 = 0; i < data->batch_size_; ++i) {
    data->minibatch_[i].fea_pulls_.assign(fea_pulls.begin() + j1, fea_pulls.begin() + j1 + data->minibatch_[i].feas_.size());
    j1 += data->minibatch_[i].feas_.size();
    CHECK(data->minibatch_[i].feas_.size() == data->minibatch_[i].fea_pulls_.size());

    data->minibatch_[i].memory_fea_pulls_.assign(memory_fea_pulls.begin() + j2, memory_fea_pulls.begin() + j2 + data->minibatch_[i].memory_feas_.size());
    j2 += data->minibatch_[i].memory_feas_.size();
    CHECK(data->minibatch_[i].memory_feas_.size() == data->minibatch_[i].memory_fea_pulls_.size());
  }
  ts2 = absl::Now();
  perf_pull_sparse_.record(ts1, ts2);
}

void RTSparseLearner::push_params(ThreadLocalData *data) {
  if (test_mode_) {
    return;
  }
  absl::Time ts1;
  absl::Time ts2;

  if (phase_ == TrainingPhase::JOINING) {
    ts1 = absl::Now();
    dense_table_client_.push(data->dnn_pushs_);
    summary_table_client_.push(data->dnn_summary_pushs_);
    ts2 = absl::Now();
    perf_push_dense_.record(ts1, ts2);
  }

  ts1 = absl::Now();
  vector<SparseFeatureVer1> feas;
  vector<SparseValueVer1> fea_pushs;
  vector<SparseFeatureVer1> memory_feas;
  vector<SparseEmbeddingVer1> memory_fea_pushs;

  for (int i = 0; i < data->batch_size_; ++i) {
    feas.insert(feas.end(), data->minibatch_[i].feas_.begin(), data->minibatch_[i].feas_.end());
    fea_pushs.insert(fea_pushs.end(), data->minibatch_[i].fea_pushs_.begin(), data->minibatch_[i].fea_pushs_.end());
    memory_feas.insert(memory_feas.end(), data->minibatch_[i].memory_feas_.begin(), data->minibatch_[i].memory_feas_.end());
    memory_fea_pushs.insert(memory_fea_pushs.end(), data->minibatch_[i].memory_fea_pushs_.begin(), data->minibatch_[i].memory_fea_pushs_.end());
  }
  sparse_table_client_.push(feas, fea_pushs);
  memory_table_client_.push(memory_feas, memory_fea_pushs);
  ts2 = absl::Now();
  perf_push_sparse_.record(ts1, ts2);
}

void RTSparseLearner::initialize_thread_local_data() {
  thread_num_ = ConfigManager::pick_local_thread_num();
  thread_local_data_.resize(thread_num_);

  for (int i = 0; i < thread_num_; ++i) {
    ps_dnn_plugin_.build_graph(&(thread_local_data_[i]));
    thread_local_data_[i].tid_ = i;
  }
}

void RTSparseLearner::finalize_thread_local_data() {
}

void RTSparseLearner::initialize_param_table() {
  dense_table_client_.create("ctr_dnn_param");
  summary_table_client_.create("ctr_dnn_summary_param");
  sparse_table_client_.create("ctr_feature");
  memory_table_client_.create("ctr_memory");
  MPIAgent::mpi_barrier_group();

  dense_table_client_.resize(ps_dnn_plugin_.tot_param_len());
  summary_table_client_.resize(ps_dnn_plugin_.tot_summary_len());
  MPIAgent::mpi_barrier_group();

  if (MPIAgent::mpi_rank_group() == 0) {
    vector<DenseValueVer1> init_dnn;
    ps_dnn_plugin_.init_dnn_param(&init_dnn);
    dense_table_client_.assign(init_dnn);

    vector<SummaryValueVer1> init_summary;
    ps_dnn_plugin_.init_summary_param(&init_summary);
    summary_table_client_.assign(init_summary);
  }
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::finalize_param_table() {
  MPIAgent::mpi_barrier_group();
}

void RTSparseLearner::save_param_table(const string& path) {
  if (MPIAgent::mpi_rank_group() == 0) {
    FSAgent::hdfs_mkdir(path + "/param");
    FSAgent::hdfs_mkdir(path + "/summary");
    FSAgent::hdfs_mkdir(path + "/feature");
    FSAgent::hdfs_mkdir(path + "/memory");
  }

  dense_table_client_.save(path + "/param");
  summary_table_client_.save(path + "/summary");
  sparse_table_client_.save(path + "/feature");
  memory_table_client_.save(path + "/memory");
}

void RTSparseLearner::init_pushs(ThreadLocalData *data) {
  int fm_dim = ConfigManager::pick_training_rule().sparse_.fm_rule_.dim_;
  int mf_dim = ConfigManager::pick_training_rule().sparse_.mf_rule_.dim_;
  int memory_dim = ConfigManager::pick_training_rule().sparse_.dic_rule_.dim_;

  for (int i = 0; i < data->batch_size_; ++i) {
    Instance& ins = data->minibatch_[i];

    for (int f = 0; f < ins.fea_num_; ++f) {
      ins.fea_pushs_[f].slot_    = ins.feas_[f].slot_;
      ins.fea_pushs_[f].version_ = ins.fea_pulls_[f].version_;
      ins.fea_pushs_[f].show_    = 0;
      ins.fea_pushs_[f].clk_     = 0;
      ins.fea_pushs_[f].lr_w_    = 0;
      ins.fea_pushs_[f].fm_w_    = 0;
      ins.fea_pushs_[f].fm_v_.assign(fm_dim, 0.0);
      ins.fea_pushs_[f].mf_w_    = 0;
      ins.fea_pushs_[f].mf_v_.assign(mf_dim, 0.0);
      ins.fea_pushs_[f].wide_w_  = 0;
    }

    // for memory dnn
    for (int f = 0; f < ins.memory_fea_num_; ++f) {
      ins.memory_fea_pushs_[f].slot_    = ins.memory_feas_[f].slot_;
      ins.memory_fea_pushs_[f].version_ = ins.memory_fea_pulls_[f].version_;
      ins.memory_fea_pushs_[f].count_   = 0;
      ins.memory_fea_pushs_[f].embedding_.assign(memory_dim, 0.0);
    }
  }
}

void RTSparseLearner::record_to_instance(const Record& rec, Instance *ins) {
  ins->lineid_ = rec.lineid_;
  CHECK(absl::SimpleAtof(ins->lineid_, &ins->adq_)) << "lineid = " << ins->lineid_;
  ins->label_  = rec.clk_;
  ins->bid_    = rec.bid_;
  ins->prior_  = 0;

  ins->position_fea_ = 0;
  ins->position_idx_ = -1;

  ins->fea_num_ = 0;
  ins->feas_.clear();
  ins->fea_pulls_.clear();
  ins->fea_pushs_.clear();

  ins->memory_fea_num_ = 0;
  ins->memory_feas_.clear();
  ins->memory_fea_pulls_.clear();
  ins->memory_fea_pushs_.clear();

  const int position_slot = ConfigManager::pick_worker_rule().position_slot_;
  const vector<uint64_t>& position_feas = ConfigManager::pick_worker_rule().position_feas_;

  for (const SparseFeatureVer1& fea : rec.feas_) {
    // base dnn
    if (base_slot_set_.get(fea.slot_) >= 0) {
      ins->feas_.push_back(fea);
      if ((int)(fea.slot_) == position_slot) {
        ins->position_fea_ = fea.sign_;
        auto it = std::find(position_feas.begin(), position_feas.end(), fea.sign_);
        if (it != position_feas.end()) {
          ins->position_idx_ = it - position_feas.begin();
        } else {
          fprintf(stderr, "Unrecognized positional feature with lineid %s and sign %llu:%u\n",
                  rec.lineid_.c_str(), (unsigned long long)fea.sign_, fea.slot_);
        }
      }
    }

    // memory dnn
    if (memory_slot_set_.get(fea.slot_) >= 0) {
      ins->memory_feas_.push_back(fea);
    }
  }

  ins->fea_num_ = (int)(ins->feas_.size());
  ins->memory_fea_num_ = (int)(ins->memory_feas_.size());

  ins->vec_values_ = rec.vec_feas_;

  // initialize pulls and pushs
  ins->fea_pulls_.resize(std::max((size_t)ins->fea_num_, ins->fea_pulls_.size()));
  ins->fea_pushs_.resize(std::max((size_t)ins->fea_num_, ins->fea_pushs_.size()));
  ins->memory_fea_pulls_.resize(std::max((size_t)ins->memory_fea_num_, ins->memory_fea_pulls_.size()));
  ins->memory_fea_pushs_.resize(std::max((size_t)ins->memory_fea_num_, ins->memory_fea_pushs_.size()));
}

void RTSparseLearner::process_data_thread(int tid, Channel<Record> in_chan) {
  ThreadLocalData *data = &(thread_local_data_[tid]);
  data->phase_ = phase_;

  absl::Time ts1;
  absl::Time ts2;

  vector<Record> buffer;
  while (in_chan->read(buffer) > 0) {
    CHECK(buffer.size() <= (size_t)ConfigManager::pick_worker_rule().batch_size_);

    data->batch_size_ = (int)buffer.size();
    data->minibatch_.resize(data->batch_size_);
    for (int i = 0; i < data->batch_size_; ++i) {
      record_to_instance(buffer[i], &(data->minibatch_[i]));
    }

    ts1 = absl::Now();
    pull_params(data);
    ts2 = absl::Now();
    perf_pull_.record(ts1, ts2);

    ts1 = absl::Now();
    preprocess(data);
    ts2 = absl::Now();
    perf_preprocess_.record(ts1, ts2);

    ts1 = absl::Now();
    feed_forward(data);
    ts2 = absl::Now();
    perf_forward_.record(ts1, ts2);

    ts1 = absl::Now();
    back_propagate(data);
    ts2 = absl::Now();
    perf_back_propagate_.record(ts1, ts2);

    ts1 = absl::Now();
    push_params(data);
    ts2 = absl::Now();
    perf_push_.record(ts1, ts2);

    ts1 = absl::Now();
    for (int i = 0; i < data->batch_size_; ++i) {
      lr_auc_.add(data->minibatch_[i].lr_pred_, data->minibatch_[i].label_);
      mf_auc_.add(data->minibatch_[i].mf_pred_, data->minibatch_[i].label_);
      fm_auc_.add(data->minibatch_[i].fm_pred_, data->minibatch_[i].label_);
      adq_auc_.add(data->minibatch_[i].adq_, data->minibatch_[i].label_);
      if (phase_ == TrainingPhase::JOINING) {
        for (int j = 0; j < dnn_auc_num_; ++j) {
          dnn_auc_[j].add(data->minibatch_[i].dnn_preds_[j], data->minibatch_[i].label_);
        }
      }
    }
    ts2 = absl::Now();
    perf_auc_.record(ts1, ts2);
  }
}

void RTSparseLearner::process_data(Channel<Record> in_chan) {
  in_chan->set_block_size(ConfigManager::pick_worker_rule().batch_size_);

  parallel_run([this, in_chan](int tid) {
    process_data_thread(tid, in_chan);
  });

  return;
}

} // namespace model
} // namespace ps

