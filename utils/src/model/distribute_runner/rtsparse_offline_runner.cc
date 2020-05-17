#include "model/distributed_runner/rtsparse_offline_runner.h"

#include <stdio.h>
#include <time.h>
#include <butil/logging.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/random/random.h"
#include "toolkit/string_agent.h"
#include "toolkit/shell_agent.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/fs_agent.h"
#include "toolkit/channel.h"
#include "toolkit/parallel_data_processor.h"
#include "model/data/record.h"

using std::vector;
using std::string;
using std::set;
using ps::toolkit::LineFileReader;
using ps::toolkit::ShellAgent;
using ps::toolkit::MPIAgent;
using ps::toolkit::FSAgent;
using ps::toolkit::Channel;
using ps::toolkit::make_channel;
using ps::toolkit::ParallelDataProcessor;
using ps::runtime::ConfigManager;

namespace ps {
namespace model {

template<class... ARGS>
static inline void rank0_fprintf(FILE *stream, const char *format, ARGS && ... args) {
  if (MPIAgent::mpi_rank_group() == 0) {
    fprintf(stream, format, args...);
  }
}
// static inline void rank0_fprintf(FILE *stream, const char *format, ...) {
//   if (MPIAgent::mpi_rank_group() == 0) {
//     va_list arg;
//     va_start(arg, format);
//     fprintf(stream, format, arg);
//     va_end(arg);
//   }
// }

void RTSparseOfflineRunner::run() {
  rank0_fprintf(stdout, "\n");

  worker_rule_ = &(ConfigManager::pick_worker_rule());
  print_worker_rule();

  learner_.initialize();
  initialize();

  // train day by day
  for (size_t day_i = 0; day_i < train_days_.size(); ++day_i) {
    Day& day = train_days_[day_i];

    rank0_fprintf(stdout, "======== Train DAY %s ========\n", day.day.c_str());
    LOG(INFO)<< "begin compute day: "<< day.day << " " << day.joining << " " << day.updating;

    // download train data
    vector<Record> train_data = get_train_data();
    Channel<Record> data_chan = make_channel<Record>();
    data_chan->write(std::move(train_data));
    data_chan->close();

    learner_.begin_day();
    learner_.set_testmode(false);
    if (day.updating) {
      rank0_fprintf(stdout, "Begin updating ...\n");
      learner_.begin_pass(TrainingPhase::UPDATING);
      learner_.process_data(data_chan);
      learner_.end_pass();
      rank0_fprintf(stdout, "End updating.\n");
    }
    if (day.joining) {
      rank0_fprintf(stdout, "Begin joining ...\n");
      learner_.begin_pass(TrainingPhase::JOINING);
      learner_.process_data(data_chan);
      learner_.end_pass();
      rank0_fprintf(stdout, "End joining.\n");
    }
    learner_.end_day();
    data_chan = NULL;

    rank0_fprintf(stdout, "Time decay ...\n");
    learner_.time_decay();
    rank0_fprintf(stdout, "Shrink table ...\n");
    learner_.shrink_table();

    if (worker_rule_->offline_worker_rule_.test_per_step_ == true){
      // download test data
      vector<Record> test_data = get_test_data();
      Channel<Record> test_chan = make_channel<Record>();
      test_chan->write(std::move(test_data));
      test_chan->close();

      learner_.set_testmode(true);
      rank0_fprintf(stdout, "Begin testing ...\n");
      learner_.begin_pass(TrainingPhase::JOINING);
      learner_.process_data(test_chan);
      learner_.end_pass();
      rank0_fprintf(stdout, "End testing.\n");
      learner_.set_testmode(false);
    }

    // learner_.print_stat();

    if (MPIAgent::mpi_rank_group() == 0) {
      if (worker_rule_->offline_worker_rule_.model_path_ != "" && model_days_.count(day.day) != 0) {
        // std::string save_model_path = worker_rule_->offline_worker_rule_.model_path_ + "/" + day.day;
        // learner_.save_model(save_model_path, "");
        // write_done_file(save_model_path, day_i);
      }
    }
  }

  if (worker_rule_->offline_worker_rule_.test_per_step_ == false
      || worker_rule_->offline_worker_rule_.dump_result_ == true || train_days_.size() == 0){
    // download test data
    vector<Record> test_data = get_test_data();
    Channel<Record> test_chan = make_channel<Record>();
    test_chan->write(std::move(test_data));
    test_chan->close();

    rank0_fprintf(stdout, "Begin testing ...\n");
    learner_.set_testmode(true);
    // learner_.set_dump_result(dump_result_, result_data_path_);
    learner_.begin_pass(TrainingPhase::JOINING);
    learner_.process_data(test_chan);
    learner_.end_pass();
    rank0_fprintf(stdout, "End testing.\n");
    // learner_.set_dump_result(false_, result_data_path_);
    learner_.set_testmode(false);
  }

  // save model
  if (train_days_.size() > 0) {
    string all_model_path = worker_rule_->offline_worker_rule_.model_path_;
    string curr_model_path = all_model_path + string("/") + train_days_[train_days_.size() - 1].day;
    if (MPIAgent::mpi_rank_group() == 0) {
      if (!FSAgent::hdfs_exists(all_model_path)) {
        FSAgent::hdfs_mkdir(all_model_path);
      }
      if (FSAgent::hdfs_exists(curr_model_path)) {
        FSAgent::hdfs_remove(curr_model_path);
      }
      FSAgent::hdfs_mkdir(curr_model_path);
    }

    rank0_fprintf(stdout, "Save model [%s] ...\n", curr_model_path.c_str());
    learner_.save_param_table(curr_model_path);
  }

  finalize();
  rank0_fprintf(stdout, "============= Complete =============\n");
}

void RTSparseOfflineRunner::print_worker_rule() {
  rank0_fprintf(stdout, "======== Job configurations ========\n");
  rank0_fprintf(stdout, "  > batch_size: %d\n",               worker_rule_->batch_size_);
  rank0_fprintf(stdout, "  > drop_feature: %s\n",             worker_rule_->drop_feature_ ? "true" : "false");
  rank0_fprintf(stdout, "  > load_balance: %s\n",             worker_rule_->load_balance_ ? "true" : "false");
  rank0_fprintf(stdout, "  > bias_slot: %d\n",                worker_rule_->bias_slot_);
  rank0_fprintf(stdout, "  > position_slot: %d\n",            worker_rule_->position_slot_);
  rank0_fprintf(stdout, "  > position_feas:");
  for (int i = 0; i < (int)worker_rule_->position_feas_.size(); ++i) {
    rank0_fprintf(stdout, " %llu", (unsigned long long)worker_rule_->position_feas_[i]);
  }
  rank0_fprintf(stdout, "\n");
  rank0_fprintf(stdout, "  > train_mode: %s\n",               worker_rule_->train_mode_.c_str());
  rank0_fprintf(stdout, "  > offline_runner:\n");
  rank0_fprintf(stdout, "    * model_path: %s\n",             worker_rule_->offline_worker_rule_.model_path_.c_str());
  rank0_fprintf(stdout, "    * model_donefile: %s\n",         worker_rule_->offline_worker_rule_.model_donefile_.c_str());
  rank0_fprintf(stdout, "    * model_converter: %s\n",        worker_rule_->offline_worker_rule_.model_converter_.c_str());
  rank0_fprintf(stdout, "    * data_path: %s\n",              worker_rule_->offline_worker_rule_.data_path_.c_str());
  rank0_fprintf(stdout, "    * data_donefile: %s\n",          worker_rule_->offline_worker_rule_.data_donefile_.c_str());
  rank0_fprintf(stdout, "    * data_converter: %s\n",         worker_rule_->offline_worker_rule_.data_converter_.c_str());
  rank0_fprintf(stdout, "    * train_days: %s\n",             worker_rule_->offline_worker_rule_.train_days_.c_str());
  rank0_fprintf(stdout, "    * test_days: %s\n",              worker_rule_->offline_worker_rule_.test_days_.c_str());
  rank0_fprintf(stdout, "    * model_days: %s\n",             worker_rule_->offline_worker_rule_.model_days_.c_str());
  rank0_fprintf(stdout, "    * start_update_days: %d\n",      worker_rule_->offline_worker_rule_.start_update_days_);
  rank0_fprintf(stdout, "    * dump_result: %s\n",            worker_rule_->offline_worker_rule_.dump_result_ ? "true" : "false");
  rank0_fprintf(stdout, "    * dump_result_path: %s\n",       worker_rule_->offline_worker_rule_.dump_result_path_.c_str());
  rank0_fprintf(stdout, "    * join_data_path: %s\n",         worker_rule_->offline_worker_rule_.join_data_path_.c_str());
  rank0_fprintf(stdout, "    * backdate: %s\n",               worker_rule_->offline_worker_rule_.backdate_ ? "true" : "false");
  rank0_fprintf(stdout, "    * backdate_days: %s\n",          worker_rule_->offline_worker_rule_.backdate_days_.c_str());
  rank0_fprintf(stdout, "    * open_sampling: %s\n",          worker_rule_->offline_worker_rule_.open_sampling_ ? "true" : "false");
  rank0_fprintf(stdout, "    * sampling_rate: %f\n",          worker_rule_->offline_worker_rule_.sampling_rate_);
  rank0_fprintf(stdout, "    * shuffle_data: %s\n",           worker_rule_->offline_worker_rule_.shuffle_data_ ? "true" : "false");
  rank0_fprintf(stdout, "    * test_per_step: %s\n",          worker_rule_->offline_worker_rule_.test_per_step_ ? "true" : "false");
  rank0_fprintf(stdout, "    * load_prior_model: %s\n",       worker_rule_->offline_worker_rule_.load_prior_model_ ? "true" : "false");
  rank0_fprintf(stdout, "    * recover_mode: %s\n",           worker_rule_->offline_worker_rule_.recover_mode_ ? "true" : "false");
  rank0_fprintf(stdout, "\n");
}

void RTSparseOfflineRunner::initialize() {
  const string& model_donefile = worker_rule_->offline_worker_rule_.model_donefile_;
  if (FSAgent::fs_exists(model_donefile)) {
    string tail_str = FSAgent::fs_tail(model_donefile);
    if (tail_str != "") {
      vector<string> info = absl::StrSplit(tail_str, '\t');
      CHECK(info.size() > 2) << "model donefile format error: " << model_donefile;
      load_model_path_ = info[0];
      CHECK(absl::SimpleAtoi(info[1], &last_train_day_)) << "last_train_day: " << info[1];
      last_train_day_str_ = info[2];
      absl::StripAsciiWhitespace(&last_train_day_str_);
    }
    rank0_fprintf(stdout, "model_done_file exist: load_model_path=%s, last_train_day=%d, last_train_day_str=%s\n",
                  load_model_path_.c_str(), last_train_day_, last_train_day_str_.c_str());
  }

  model_days_ = parse_day_set(worker_rule_->offline_worker_rule_.model_days_);

  const bool& load_prior_model = worker_rule_->offline_worker_rule_.load_prior_model_;
  const bool& recover_mode = worker_rule_->offline_worker_rule_.recover_mode_;

  if (load_model_path_ != "" && (load_prior_model == true || recover_mode == true) ) {
    rank0_fprintf(stdout, "begin load model: load_prior_model=%d, recover_mode=%d, load_model_path=%s\n",
                  load_prior_model, recover_mode, load_model_path_.c_str());
    // learner_.load_model(load_model_path_, model_converter); // to be continue
    // learner_.print_stat();                                  // to be continue
  }

  // remove trained days
  vector<string> train_days = parse_day_list(worker_rule_->offline_worker_rule_.train_days_);
  if (recover_mode == true){
    LOG(INFO) <<"recover_mode: true, get last day: " << last_train_day_ << " " << last_train_day_str_;
    train_days.erase(train_days.begin(), train_days.begin() + last_train_day_);
    CHECK(last_train_day_str_ == "" || train_days[0] == last_train_day_str_);
    rank0_fprintf(stdout, "recover_mode: last_train_day_str_=%s, begin_train=%s\n",
                  last_train_day_str_.c_str(), train_days[0].c_str());
  } else {
    last_train_day_ = 0;
  }

  // set train days
  train_days_.resize(train_days.size());
  for (int i = 0; i < (int)train_days.size(); i++) {
    train_downloader_.add_task(
      ShellAgent::shell_get_command_output(absl::StrFormat("DAY=%s; echo -n %s", train_days[i].c_str(),
                                           worker_rule_->offline_worker_rule_.data_path_.c_str())),
      worker_rule_->offline_worker_rule_.data_converter_,
      ShellAgent::shell_get_command_output(absl::StrFormat("DAY=%s; echo -n %s", train_days[i].c_str(),
                                           worker_rule_->offline_worker_rule_.data_donefile_.c_str()))
    );
    train_days_[i].day = train_days[i];
    train_days_[i].joining = true;
    train_days_[i].updating = false;
  }

  // set test days
  vector<string> test_days = parse_day_list(worker_rule_->offline_worker_rule_.test_days_);
  test_days_.resize(test_days.size());
  for (int i = 0; i < (int)test_days.size(); ++i) {
    test_downloader_.add_task(
      ShellAgent::shell_get_command_output(absl::StrFormat("DAY=%s; echo -n %s", test_days[i].c_str(),
                                           worker_rule_->offline_worker_rule_.data_path_.c_str())),
      worker_rule_->offline_worker_rule_.data_converter_,
      ShellAgent::shell_get_command_output(absl::StrFormat("DAY=%s; echo -n %s", test_days[i].c_str(),
                                           worker_rule_->offline_worker_rule_.data_donefile_.c_str()))
    );
    test_days_[i].day = test_days[i];
    test_days_[i].joining = false;
    test_days_[i].updating = false;
  }
}

vector<Record> RTSparseOfflineRunner::get_train_data() {
  static absl::BitGen gen;
  if (!train_downloader_.next_task()) {
    LOG(FATAL) << "no training data left.";
  }

  rank0_fprintf(stdout, "Downloading train data from: %s\n", train_downloader_.task_info().path.c_str());
  Channel<string> in_chan = train_downloader_.get_data();
  LOG(INFO) << "getting data finished, size = " << in_chan->size() << ".";

  Channel<Record> rec_chan = ParallelDataProcessor::run_map<Record>(in_chan, [](const std::string& line) {
    thread_local Record rec;
    parse_record(line.c_str(), rec);
    return rec;
  });
  LOG(INFO) << "parsing data finished, size = " << rec_chan->size() << ".";

  const float& sampling_rate = worker_rule_->offline_worker_rule_.sampling_rate_;
  vector<Record> train_data;
  vector<Record> recs;
  rec_chan->read_all(recs);
  for (size_t i = 0; i < recs.size(); ++i){
    Record rec = std::move(recs[i]);
    int show = rec.show_;
    int clk = rec.clk_;

    for (int k = 0; k < show; k++) {
      rec.show_ = 1;
      rec.clk_ = int(k < clk);

      if (worker_rule_->offline_worker_rule_.open_sampling_ == false){
        train_data.push_back(rec);
      } else {
        if (rec.clk_ == 1) {
          train_data.push_back(rec);
        } else {
          bool sample = false;
          if (absl::uniform_real_distribution<float>(0, 1)(gen) <= sampling_rate) {
            sample = true;
          }
          if (sample == true) {
            train_data.push_back(rec);
          }
        }
      }
    }
  }
  LOG(INFO) << "prerocessing data finished, size = " << train_data.size() << ".";
  if (worker_rule_->offline_worker_rule_.shuffle_data_){
    rank0_fprintf(stdout, "shuffling data\n");
    std::shuffle(train_data.begin(), train_data.end(), gen);
  }

  print_data_size((long long)train_data.size());
  MPIAgent::mpi_barrier_group();

  return std::move(train_data);
}

vector<Record> RTSparseOfflineRunner::get_test_data() {
  vector<Record> test_data;

  test_downloader_.restart();
  while (test_downloader_.next_task()) {
    rank0_fprintf(stdout, "Downloading test data from: %s\n", test_downloader_.task_info().path.c_str());
    Channel<string> in_chan = test_downloader_.get_data();
    LOG(INFO) << "getting data finished, size = " << in_chan->size() << ".";
    Channel<Record> rec_chan = ParallelDataProcessor::run_map<Record>(in_chan, [](const string& line) {
      thread_local Record rec;
      parse_record(line.c_str(), rec);
      return rec;
    });
    LOG(INFO) << "parsing data finished, size = " << rec_chan->size() << ".";

    vector<Record> recs;
    rec_chan->read_all(recs);

    for (size_t i = 0; i < recs.size(); ++i){
      Record rec = std::move(recs[i]);
      int show = rec.show_;
      int clk = rec.clk_;
      for (int k = 0; k < show; k++) {
        rec.show_ = 1;
        rec.clk_ = int(k < clk);
        test_data.push_back(rec);
      }
    }
  }
  LOG(INFO) << "preprocessing all test data finished, size = " << test_data.size();

  print_data_size((long long)test_data.size());
  MPIAgent::mpi_barrier_group();

  return std::move(test_data);
}

void RTSparseOfflineRunner::finalize() {
  learner_.finalize();
}

vector<string> RTSparseOfflineRunner::parse_day_list(const string& str) {
  return std::move(absl::StrSplit(ShellAgent::shell_get_command_output(absl::StrFormat("echo -n %s", str.c_str())), ' '));
}

set<string> RTSparseOfflineRunner::parse_day_set(const string& str) {
  set<string> p;
  for (auto s : parse_day_list(str)) {
    p.insert(s);
  }
  return std::move(p);
}

void RTSparseOfflineRunner::print_data_size(long long local_size) {
  long long total   = MPIAgent::mpi_allreduce_group(local_size, MPI_SUM);
  long long average = total / MPIAgent::mpi_size_group();
  long long max     = MPIAgent::mpi_allreduce_group(local_size, MPI_MAX);
  rank0_fprintf(stdout, "Instance count: %lld, Avg instance count: %lld, Max instance count: %lld\n", total, average, max);
}

void RTSparseOfflineRunner::write_done_file(const string& model_path, int i_day) {
}

} // namespace model
} // namespace ps

