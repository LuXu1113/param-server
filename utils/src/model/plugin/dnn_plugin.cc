#include "model/plugin/ps_dnn_plugin.h"

#include "absl/random/random.h"
#include "model/data/component.h"
#include "model/data/matrix_output.h"
#include "model/tool/eigen_impl.h"
#include "model/layers.h"

using std::map;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using ps::runtime::ConfigManager;
using ps::runtime::TrainingRule;
using ps::param_table::DenseValueVer1;
using ps::param_table::DenseValueVer1Pull;
using ps::param_table::DenseValueVer1Push;
using ps::param_table::SummaryValueVer1;
using ps::param_table::SparseFeatureVer1;
using ps::param_table::SparseKeyVer1;
using ps::param_table::SparseValueVer1;
using ps::param_table::SparseEmbeddingVer1;

namespace ps {
namespace model {

template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params) {
  return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

ELSDNNPlugin::ELSDNNPlugin() {}

void ELSDNNPlugin::get_pull_dense(const vector<DenseValueVer1Pull>& data, vector<shared_ptr<MatrixOutput> > *params) {
  CHECK(data.size() == (size_t)tot_param_len_);
  CHECK((*params).size() == (size_t)param_num_);

  int offset = 0;
  for (int i = 0; i < param_num_; offset += params_[i++].length_) {
    const DenseValueVer1Pull *source = &(data[offset]);
    (*params)[i]->value().resize(params_[i].rown_, params_[i].coln_);
    float *dest = (*params)[i]->value().data();
    for (int j = 0; j < params_[i].length_; ++j) {
      dest[j] = source[j].weight_;
    }
  }
  CHECK((uint64_t)offset == tot_param_len_);
}

void ELSDNNPlugin::set_push_dense(vector<DenseValueVer1Push> *data, const vector<shared_ptr<MatrixOutput> >& params) {
  CHECK(params.size() == (size_t)param_num_);
  data->resize(tot_param_len_);

  int offset = 0;
  for (int i = 0; i < param_num_; offset += params_[i++].length_) {
    DenseValueVer1Push *dest = &((*data)[offset]);
    if (!(params[i]->has_gradient())) {
      for (int j = 0; j < params_[i].length_; ++j) {
        dest[j].weight_ = NAN;
        dest[j].norm_grad_ = NAN;
        dest[j].norm_weight_ = NAN;
      }
    } else {
      CHECK(matrix_size_equals(params[i]->gradient(), params_[i].rown_, params_[i].coln_));
      const float *source = params[i]->gradient().data();
      for (int j = 0; j < params_[i].length_; ++j) {
        dest[j].weight_ = source[j];
        dest[j].norm_grad_ = 0;
        dest[j].norm_weight_ = 0;
      }
    }
  }
  CHECK((uint64_t)offset == tot_param_len_);
}

void ELSDNNPlugin::get_pull_summaries(const vector<SummaryValueVer1>& data, vector<shared_ptr<MatrixOutput> > *summaries) {
  CHECK(data.size() == (size_t)tot_summary_len_);
  CHECK((*summaries).size() == (size_t)summary_num_);

  int offset = 0;
  for (int i = 0; i < summary_num_; offset += summaries_[i++].length_) {
    const SummaryValueVer1 *source = &(data[offset]);
    (*summaries)[i]->value().resize(3, summaries_[i].length_);
    Eigen::MatrixXf& dest = (*summaries)[i]->value();
    for (int j = 0; j < summaries_[i].length_; ++j) {
      dest(0, j) = source[j].n_;
      dest(1, j) = source[j].sum_;
      dest(2, j) = source[j].squared_sum_;
    }
  }
  CHECK((uint64_t)offset == tot_summary_len_);
}

void ELSDNNPlugin::set_push_summaries(vector<SummaryValueVer1> *data, const vector<shared_ptr<MatrixOutput> >& summaries) {
  CHECK(summaries.size() == (size_t)summary_num_);
  data->resize(tot_summary_len_);

  int offset = 0;
  for (int i = 0; i < summary_num_; offset += summaries_[i++].length_) {
    SummaryValueVer1 *dest = &((*data)[offset]);
    if (!(summaries[i]->has_gradient())) {
      continue;
    } else {
      const Eigen::MatrixXf& source = summaries[i]->gradient();
      CHECK(matrix_size_equals(source, 3, summaries_[i].length_));
      for (int j = 0; j < summaries_[i].length_; ++j) {
        dest[j].n_           = source(0, j);
        dest[j].sum_         = source(1, j);
        dest[j].squared_sum_ = source(2, j);
      }
    }
  }
  CHECK((uint64_t)offset == tot_summary_len_);
}

void ELSDNNPlugin::sparse_feature_to_tensor(ThreadLocalData *data) {
  // bias input
  data->dnn_bias_input_->value().setOnes(data->batch_size_, 1);

  // position feature input
  int position_feature_num = ConfigManager::pick_worker_rule().position_feas_.size();

  if (position_feature_num > 0) {
    // fixed position feature
    Eigen::MatrixXf& fixed_pos_mat = data->dnn_fixed_position_input_->value();
    fixed_pos_mat.setZero(data->batch_size_, position_feature_num);
    fixed_pos_mat.col(0).setOnes();

    // position feature
    Eigen::MatrixXf& pos_mat = data->dnn_position_input_->value();
    pos_mat.setZero(data->batch_size_, position_feature_num);
    for (int i = 0; i < data->batch_size_; ++i) {
      int idx = data->minibatch_[i].position_idx_;
      if (idx >= 0) {
        CHECK(idx < position_feature_num);
        pos_mat(i, idx) = 1;
      }
    }
  } else {
    LOG(WARNING) << "no position feature.";
  }

  // sparse plugins input
  int fm_dim = ConfigManager::pick_training_rule().sparse_.fm_rule_.dim_;
  int mf_dim = ConfigManager::pick_training_rule().sparse_.mf_rule_.dim_;

  Eigen::MatrixXf& show_mat      = data->dnn_show_input_->value();
  Eigen::MatrixXf& clk_mat       = data->dnn_clk_input_->value();
  Eigen::MatrixXf& lr_mat        = data->dnn_lr_input_->value();
  Eigen::MatrixXf& fm_mat        = data->dnn_fm_input_->value();
  Eigen::MatrixXf& mf_mat        = data->dnn_mf_input_->value();

  show_mat.setZero(data->batch_size_, base_slot_num_);
  clk_mat.setZero(data->batch_size_, base_slot_num_);
  lr_mat.setZero(data->batch_size_, base_slot_num_);
  fm_mat.setZero(data->batch_size_, base_slot_num_ * (fm_dim + 1));
  mf_mat.setZero(data->batch_size_, base_slot_num_ * (mf_dim + 1));

  // memory input
  int dic_dim = ConfigManager::pick_training_rule().sparse_.dic_rule_.dim_;
  Eigen::MatrixXf& memory_mat = data->dnn_memory_input_->value();
  memory_mat.setZero(data->batch_size_, memory_slot_num_ * dic_dim);

  // vector input
  for (auto& iter : data->vec_inputs_) {
    iter.second.vinput_->value().setZero(data->batch_size_, iter.second.dim_);
  }

  for (int i = 0; i < data->batch_size_; ++i) {
    // fill cvm input
    int fea_num = data->minibatch_[i].fea_num_;
    const SparseFeatureVer1* feas    = &(data->minibatch_[i].feas_[0]);
    const SparseValueVer1* fea_pulls = &(data->minibatch_[i].fea_pulls_[0]);
    for (int j = 0; j < fea_num; ++j) {
      const SparseValueVer1& fea_pull = fea_pulls[j];
      int idx = base_slot_mapping_.get(feas[j].slot_);
      if (idx >= 0) {
        show_mat(i, idx) += fea_pull.show_;
        clk_mat(i, idx)  += fea_pull.clk_;
        lr_mat(i, idx)   += fea_pull.lr_w_;

        fm_mat(i, idx * (fm_dim + 1)) += fea_pull.fm_w_;
        if (!(fea_pull.fm_v_.empty())) {
          CHECK((int)fea_pull.fm_v_.size() == fm_dim);
          for (int k = 0; k < fm_dim; ++k) {
            fm_mat(i, idx * (fm_dim + 1) + k + 1) += fea_pull.fm_v_[k];
          }
        }

        mf_mat(i, idx * (mf_dim + 1)) += fea_pull.mf_w_;
        if (!(fea_pull.mf_v_.empty())) {
          CHECK((int)fea_pull.mf_v_.size() == mf_dim);
          for (int k = 0; k < mf_dim; ++k) {
            mf_mat(i, idx * (mf_dim + 1) + k + 1) += fea_pull.mf_v_[k];
          }
        }
      }
    }

    // fill memory input
    const SparseFeatureVer1   *memory_feas  = &(data->minibatch_[i].memory_feas_[0]);
    const SparseEmbeddingVer1 *memory_pulls = &(data->minibatch_[i].memory_fea_pulls_[0]);

    int memory_fea_num = data->minibatch_[i].memory_fea_num_;
    for (int j = 0; j < memory_fea_num; ++j) {
      const SparseEmbeddingVer1& memory_pull = memory_pulls[j];
      int idx = memory_slot_mapping_.get(memory_feas[j].slot_);
      if (idx >= 0) {
        if(!(memory_pull.embedding_.empty())) {
          CHECK((int)memory_pull.embedding_.size() == dic_dim);
          for (int k = 0; k < dic_dim; ++k) {
            memory_mat(i, idx * dic_dim + k) += memory_pull.embedding_[k];
          }
        }
      }
    }

    // fill vector input
    for (auto& iter : data->vec_inputs_) {
      map<string, vector<float> >& vv = data->minibatch_[i].vec_values_;
      int dim = iter.second.dim_;
      if (vv.find(iter.first) != vv.end()) {
        Eigen::MatrixXf& mat = iter.second.vinput_->value();
        for (int k = 0; k < dim && k < (int)vv[iter.first].size(); ++k) {
          mat(i, k) = vv[iter.first][k];
        }
      }
    }
  }

  show_mat = (show_mat.array() + 1).log();
  clk_mat  = (clk_mat.array() + 1).log();
  data->dnn_ctr_input_->value() = clk_mat - show_mat;

  // LOG(INFO) << "show_mat: " << show_mat;
  // LOG(INFO) << "clk_mat: " << clk_mat;
  // LOG(INFO) << "lr_mat: " << lr_mat;
  // LOG(INFO) << "mf_mat: " << mf_mat;
}

void ELSDNNPlugin::tensor_to_sparse_grad(ThreadLocalData *data) {
  // sparse plugins input
  int fm_dim = ConfigManager::pick_training_rule().sparse_.fm_rule_.dim_;
  int mf_dim = ConfigManager::pick_training_rule().sparse_.mf_rule_.dim_;

  Eigen::MatrixXf& lr_grad = data->dnn_lr_input_->gradient();
  Eigen::MatrixXf& fm_grad = data->dnn_fm_input_->gradient();
  Eigen::MatrixXf& mf_grad = data->dnn_mf_input_->gradient();

  // memory dnn
  int dic_dim = ConfigManager::pick_training_rule().sparse_.dic_rule_.dim_;
  Eigen::MatrixXf& memory_input_grad = data->dnn_memory_input_->gradient();

  for (int i = 0; i < data->batch_size_; ++i) {
    // cvm grad
    int fea_num = data->minibatch_[i].fea_num_;
    const SparseFeatureVer1 *feas = &(data->minibatch_[i].feas_[0]);
    SparseValueVer1 *fea_pushs = &(data->minibatch_[i].fea_pushs_[0]);
    for (int j = 0; j < fea_num; ++j) {
      SparseValueVer1& fea_push = fea_pushs[j];
      int idx = base_slot_mapping_.get(feas[j].slot_);
      if (idx >= 0) {
        if (data->dnn_lr_input_->has_gradient()) {
          fea_push.lr_w_ += lr_grad(i, idx);
        }
        if (data->dnn_fm_input_->has_gradient()) {
          fea_push.fm_w_ += fm_grad(i, idx * (fm_dim + 1));
          if(!(fea_push.fm_v_.empty())) {
            for (int k = 0; k < fm_dim; ++k) {
              fea_push.fm_v_[k] += fm_grad(i, idx * (fm_dim + 1) + k + 1);
            }
          }
        }
        if (data->dnn_mf_input_->has_gradient()) {
          fea_push.mf_w_ += mf_grad(i, idx * (mf_dim + 1));
          if(!fea_push.mf_v_.empty()) {
            for (int k = 0; k < mf_dim; ++k) {
              fea_push.mf_v_[k] += mf_grad(i, idx * (mf_dim + 1) + k + 1);
            }
          }
        }
      }
    }

    // memory grad
    int memory_fea_num = data->minibatch_[i].memory_fea_num_;
    const SparseFeatureVer1 *memory_feas = &(data->minibatch_[i].memory_feas_[0]);
    SparseEmbeddingVer1 *memory_pushs    = &(data->minibatch_[i].memory_fea_pushs_[0]);
    for (int j = 0; j < memory_fea_num; ++j) {
      SparseEmbeddingVer1& memory_push = memory_pushs[j];
      int idx = memory_slot_mapping_.get(memory_feas[j].slot_);
      if(idx >= 0){
        if((data->dnn_memory_input_->has_gradient())) {
          memory_push.count_ += 1.0;
          if(!(memory_push.embedding_.empty())) {
            for (int k = 0; k < dic_dim; ++k) {
              memory_push.embedding_[k] += memory_input_grad(i, idx * dic_dim + k);
            }
          }
        }
      }
    }
  }
}

void ELSDNNPlugin::initialize() {
  CHECK(!is_inited_);

  LayerFactory::init();
  const TrainingRule& rule = ConfigManager::pick_training_rule();

  // base dnn
  base_slot_num_ = rule.sparse_.base_slots_.size();
  for (int i = 0; i < base_slot_num_; ++i) {
    base_slot_mapping_.set(rule.sparse_.base_slots_[i], i);
  }

  // space time dnn
  addition_slot_num_ = rule.sparse_.addition_slots_.size();
  for (int i = 0; i < addition_slot_num_; ++i) {
    addition_slot_mapping_.set(rule.sparse_.addition_slots_[i], i);
  }

  // memory dnn
  memory_slot_num_ = rule.sparse_.memory_slots_.size();
  for (int i = 0; i < memory_slot_num_; ++i) {
    memory_slot_mapping_.set(rule.sparse_.memory_slots_[i], i);
  }

  // gaussian outputs
  use_gaussian_   = rule.dense_.use_gaussian_;
  gaussian_names_ = rule.dense_.gaussian_output_names_;
  gaussian_num_   = gaussian_names_.size();

  // memory outpus
  memory_output_names_ = rule.dense_.memory_output_names_;
  memory_output_num_   = memory_output_names_.size();

  // wide plugin
  use_wide_ = rule.dense_.use_wide_;

  // output q
  q_names_  = rule.dense_.q_names_;
  q_num_    = q_names_.size();
  q_weight_ = rule.dense_.q_weight_;
  q_weight_num_ = q_weight_.size();
  CHECK(q_num_ == q_weight_num_);

  back_propagate_input_ = rule.dense_.back_propagate_input_;
  back_addition_input_  = rule.dense_.back_addition_input_;
  back_memory_input_    = rule.dense_.back_memory_input_;

  // dnn params
  param_num_ = rule.dense_.param_.size();
  params_.resize(param_num_);
  tot_param_len_ = 0;

  for (int i = 0; i < param_num_; ++i) {
    params_[i].name_   = rule.dense_.param_[i].name_;
    params_[i].rown_   = rule.dense_.param_[i].row_n_;
    params_[i].coln_   = rule.dense_.param_[i].col_n_;
    params_[i].length_ = params_[i].rown_ * params_[i].coln_;
    params_[i].init_range_ = rule.dense_.param_[i].init_range_
                           * rule.dense_.global_init_range_
                           / ((rule.dense_.param_[i].scale_by_row_n_) ? (sqrt((float)params_[i].rown_)) : (1.0));

    tot_param_len_ += params_[i].length_;
  }

  // dnn summarys
  summary_num_ = rule.dense_.summary_.size();
  summaries_.resize(summary_num_);
  tot_summary_len_ = 0;

  summary_decay_rate_          = rule.dense_.summary_decay_rate_;
  summary_squared_sum_epsilon_ = rule.dense_.summary_squared_sum_epsilon_;

  for (int i = 0; i < summary_num_; ++i) {
    summaries_[i].name_   = rule.dense_.summary_[i].name_;
    summaries_[i].length_ = rule.dense_.summary_[i].length_;
    tot_summary_len_ += summaries_[i].length_;
  }

  // dnn layers
  vector<string> test_joining   = rule.dense_.test_layers_at_joining_;
  vector<string> train_joining  = rule.dense_.train_layers_at_joining_;
  vector<string> test_updating  = rule.dense_.test_layers_at_updating_;
  vector<string> train_updating = rule.dense_.train_layers_at_updating_;

  layer_num_ = rule.dense_.layer_conf_.size();
  layers_.resize(layer_num_);
  for (int i = 0; i < layer_num_; ++i) {
    string tag = rule.dense_.layer_conf_[i]["tag"].as<string>();
    layers_[i].test_at_joining   = find(test_joining.begin(), test_joining.end(), tag) != test_joining.end();
    layers_[i].train_at_joining  = find(train_joining.begin(), train_joining.end(), tag) != train_joining.end();
    layers_[i].test_at_updating  = find(test_updating.begin(), test_updating.end(), tag) != test_updating.end();
    layers_[i].train_at_updating = find(train_updating.begin(), train_updating.end(), tag) != train_updating.end();

    CHECK(layers_[i].test_at_joining || !layers_[i].train_at_joining);
    CHECK(layers_[i].test_at_updating || !layers_[i].train_at_updating);
  }

  is_inited_ = true;
}

void ELSDNNPlugin::build_graph(ThreadLocalData *data) {
  CHECK(is_inited_);

  const TrainingRule& rule = ConfigManager::pick_training_rule();
  ComponentTable comp_table;

  data->dnn_params_.resize(param_num_);
  for (int i = 0; i < param_num_; ++i) {
    // LOG(INFO) << "local thread "<< data->tid_ << "add_component: " << params_[i].name;
    data->dnn_params_[i] = make_shared<MatrixOutput>();
    comp_table.add_component(params_[i].name_, data->dnn_params_[i]);
  }

  data->dnn_summaries_.resize(summary_num_);
  for (int i = 0; i < summary_num_; ++i) {
    // LOG(INFO) << "local thread " << data->tid_ << "add_component: " << summaries_[i].name;
    data->dnn_summaries_[i] = std::make_shared<MatrixOutput>();
    comp_table.add_component(summaries_[i].name_, data->dnn_summaries_[i]);
  }

  comp_table.add_component("linear", make_shared<LinearActivationFunction>());
  comp_table.add_component("relu", make_shared<ReluActivationFunction>());
  comp_table.add_component("sigmoid", make_shared<SigmoidActivationFunction>());
  comp_table.add_component("tanh", make_shared<TanhActivationFunction>());

  comp_table.add_component("bias_input", data->dnn_bias_input_);
  comp_table.add_component("position_input", data->dnn_position_input_);
  comp_table.add_component("fixed_position_input", data->dnn_fixed_position_input_);
  comp_table.add_component("show_input", data->dnn_show_input_);
  comp_table.add_component("clk_input", data->dnn_clk_input_);
  comp_table.add_component("ctr_input", data->dnn_ctr_input_);
  comp_table.add_component("lr_input", data->dnn_lr_input_);
  comp_table.add_component("fm_expand_input", data->dnn_fm_input_);
  comp_table.add_component("mf_expand_input", data->dnn_mf_input_);
  comp_table.add_component("memory_input", data->dnn_memory_input_);

  int vlen = rule.dense_.vec_input_.size();
  for (int i = 0; i < vlen; ++i) {
    string name = rule.dense_.vec_input_[i].name_;
    data->vec_inputs_[name].vinput_ = make_shared<MatrixOutput>();
    data->vec_inputs_[name].dim_    = rule.dense_.vec_input_[i].dim_;
    data->vec_inputs_[name].name_   = name;
    comp_table.add_component(name, data->vec_inputs_[name].vinput_);
  }

  int addition_len = rule.dense_.addition_input_.size();
  for (int i = 0; i < addition_len; ++i) {
    std::string name = rule.dense_.addition_input_[i].name_;
    data->addition_inputs_[name].vinput_    = make_shared<MatrixOutput>();
    data->addition_inputs_[name].maskinput_ = make_shared<MatrixOutput>();
    data->addition_inputs_[name].dim_       = rule.dense_.addition_input_[i].dim_;
    data->addition_inputs_[name].vlen_      = rule.dense_.addition_input_[i].vlen_;
    data->addition_inputs_[name].name_      = name;
    comp_table.add_component(name, data->addition_inputs_[name].vinput_);
    comp_table.add_component(name + "_mask", data->addition_inputs_[name].maskinput_);
  }

  data->dnn_output_.resize(q_num_);
  for (int i = 0; i < q_num_; ++i) {
    // LOG(INFO) << "local thread " << data->tid << "add_component: " << q_names_[i];
    data->dnn_output_[i] = make_shared<MatrixOutput>();
    comp_table.add_component(q_names_[i], data->dnn_output_[i]);
  }

  data->memory_output_.resize(memory_output_num_);
  for (int i = 0; i < memory_output_num_; ++i) {
    // LOG(INFO) << "local thread " << data->tid << "add_component: " << memory_output_names_[i];
    data->memory_output_[i] = make_shared<MatrixOutput>();
    comp_table.add_component(memory_output_names_[i], data->memory_output_[i]);
  }

  if (use_gaussian_) {
    data->gaussian_output_.resize(gaussian_num_);
    for (int i = 0; i < gaussian_num_; ++i) {
      // LOG(INFO) << "local thread "<< data->tid << "add_component: " << gaussian_names_[i];
      data->gaussian_output_[i] = make_shared<MatrixOutput>();
      comp_table.add_component(gaussian_names_[i], data->gaussian_output_[i]);
    }
  }

  data->dnn_layers_ = comp_table.load_components<Layer>(rule.dense_.all_layers_conf_);

  for (int i = 0; i < (int)(data->dnn_layers_.size()); ++i) {
    data->dnn_layers_[i]->initialize();
  }

  data->dnn_preds_.resize(q_num_);
}
void ELSDNNPlugin::init_dnn_param(vector<DenseValueVer1> *init_w) {
  static absl::BitGen gen;
  CHECK(is_inited_);

  vector<float> init_ranges;
  for (int i = 0; i < param_num_; ++i) {
    for (int j = 0; j < params_[i].length_; ++j) {
      init_ranges.push_back(params_[i].init_range_);
    }
  }

  init_w->resize(tot_param_len_);
  for (uint64_t i = 0; i < tot_param_len_; ++i) {
    memset(&((*init_w)[i]), 0, sizeof((*init_w)[i]));
    (*init_w)[i].weight_ = (float)(absl::gaussian_distribution<float>(0, 1)(gen) * init_ranges[i]);
  }
}

void ELSDNNPlugin::init_summary_param(vector<SummaryValueVer1> *init_s) {
  CHECK(is_inited_);

  const TrainingRule& rule = ConfigManager::pick_training_rule();
  SummaryValueVer1 init_summary = {rule.dense_.summary_init_n_, 0, rule.dense_.summary_init_squared_sum_};

  init_s->resize(tot_summary_len_);
  for (uint64_t i = 0; i < tot_summary_len_; ++i) {
    (*init_s)[i] = init_summary;
  }
}

void ELSDNNPlugin::feed_forward(ThreadLocalData *data) {
  get_pull_dense(data->dnn_pulls_, &(data->dnn_params_));
  get_pull_summaries(data->dnn_summary_pulls_, &(data->dnn_summaries_));

  for (int i = 0; i < param_num_; ++i) {
    data->dnn_params_[i]->has_gradient()  = false;
    data->dnn_params_[i]->need_gradient() = true;
  }

  for (int i = 0; i < summary_num_; ++i) {
    data->dnn_summaries_[i]->has_gradient() = false;
    data->dnn_summaries_[i]->need_gradient() = true;
  }

  data->dnn_lr_input_->has_gradient()  = false;
  data->dnn_lr_input_->need_gradient() = back_propagate_input_;

  data->dnn_fm_input_->has_gradient()  = false;
  data->dnn_fm_input_->need_gradient() = back_propagate_input_;

  data->dnn_mf_input_->has_gradient()  = false;
  data->dnn_mf_input_->need_gradient() = back_propagate_input_;

  data->dnn_memory_input_->has_gradient()  = false;
  data->dnn_memory_input_->need_gradient() = back_memory_input_;

  // dnn output
  for (int i = 0; i < q_num_; ++i) {
    data->dnn_output_[i]->value().setZero(data->batch_size_, 1);
  }

  // memory output
  for (int i = 0; i < memory_output_num_; ++i) {
    data->memory_output_[i]->value().setZero(data->batch_size_, 1);
  }

  // gaussion output
  for (int i = 0; i< gaussian_num_; i++) {
    data->gaussian_output_[i]->value().setZero(data->batch_size_, 1);
  }

  // set dnn input component matrix
  sparse_feature_to_tensor(data);

  // layer forward
  for (int i = 0; i < layer_num_; i++) {
    data->dnn_layers_[i]->feed_forward();
  }

  for (int i = 0; i < q_num_; i++) {
    CHECK(data->dnn_output_[i]->value().cols() == 1 && data->dnn_output_[i]->value().rows() == data->batch_size_);
    if (use_wide_) {
      Eigen::MatrixXf& dnn_output_mat = data->dnn_output_[i]->value();
      for(int j = 0; j < data->batch_size_; ++j) {
        dnn_output_mat(j, 0) += data->minibatch_[j].wide_output_;
      }
    }
    data->dnn_preds_[i] = 1.0 / (1.0 + (-(data->dnn_output_[i]->value()).array()).exp());
  }

  for (int i = 0; i < data->batch_size_; ++i){
    data->minibatch_[i].dnn_preds_.resize(q_num_);
    for (int j = 0; j < q_num_; ++j) {
      data->minibatch_[i].dnn_preds_[j] = data->dnn_preds_[j](i, 0);
    }
  }
}

void ELSDNNPlugin::back_propagate(ThreadLocalData *data) {
  if (layer_num_ == 0) {
    return;
  }

  // dnn output grad
  for (int i = 0; i < q_num_; ++i) {
    data->dnn_output_[i]->has_gradient() = true;
    data->dnn_output_[i]->gradient().resize(data->batch_size_, 1);

    for (int j = 0; j < data->batch_size_; ++j) {
      data->dnn_output_[i]->gradient()(j, 0) = q_weight_[i] * (data->minibatch_[j].label_ - data->minibatch_[j].dnn_preds_[i]);
    }
  }

  // memory output grad
  for (int i = 0; i < memory_output_num_; ++i) {
    data->memory_output_[i]->has_gradient() = true;
    data->memory_output_[i]->gradient().resize(data->batch_size_, 1);

    for (int j = 0; j < data->batch_size_; ++j) {
      data->memory_output_[i]->gradient()(j, 0) = data->minibatch_[j].label_ - data->minibatch_[j].dnn_preds_[i];
    }
  }

  if (use_wide_) {
    for (int i = 0; i < data->batch_size_; ++i) {
      data->minibatch_[i].wide_grad_ = data->minibatch_[i].label_ - data->minibatch_[i].dnn_preds_[0];
    }
  }

  for (int i = layer_num_ - 1; i >= 0; --i) {
    if (layers_[i].train_at_joining) {
      data->dnn_layers_[i]->back_propagate();
    }
  }

  if (back_propagate_input_) {
    tensor_to_sparse_grad(data);
  }

  set_push_dense(&(data->dnn_pushs_), data->dnn_params_);
  float g_scale = 1.0 / data->batch_size_;
  for (uint64_t i = 0; i < tot_param_len_; ++i) {
    data->dnn_pushs_[i].weight_ *= g_scale;
  }

  set_push_summaries(&(data->dnn_summary_pushs_), data->dnn_summaries_);
}

} // namespace model
} // namespace ps

