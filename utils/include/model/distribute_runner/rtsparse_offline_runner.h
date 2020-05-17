#ifndef UTILS_INCLUDE_DISTRIBUTED_RUNNER_RTSPARSE_OFFLINE_RUNNER_H_
#define UTILS_INCLUDE_DISTRIBUTED_RUNNER_RTSPARSE_OFFLINE_RUNNER_H_

#include <vector>
#include <string>
#include <set>
#include "runtime/config_manager.h"
#include "model/distributed_learner/rtsparse_learner.h"
#include "model/data/record.h"
#include "model/tool/data_downloader.h"

namespace ps {
namespace model {

class RTSparseOfflineRunner {
 public:
  RTSparseOfflineRunner() = default;
  RTSparseOfflineRunner(const RTSparseOfflineRunner&) = delete;
  ~RTSparseOfflineRunner() = default;

  void run();

 private:
  struct Day {
    std::string day;
    bool joining;
    bool updating;
  };
  std::vector<Day> train_days_;
  std::vector<Day> test_days_;
  std::set<std::string> model_days_;

  RTSparseLearner learner_;
  const ps::runtime::WorkerRule *worker_rule_;

  std::string load_model_path_;
  int last_train_day_;
  std::string last_train_day_str_;

  DataDownloader train_downloader_;
  DataDownloader test_downloader_;

  void initialize();
  void finalize();
  std::vector<Record> get_train_data();
  std::vector<Record> get_test_data();
  std::vector<std::string> parse_day_list(const std::string& str);
  std::set<std::string> parse_day_set(const std::string& str);
  void print_data_size(const long long local_size);
  void print_worker_rule();
  void write_done_file(const std::string& model_path, int i_day);
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_DISTRIBUTED_RUNNER_RTSPARSE_OFFLINE_RUNNER_H_

