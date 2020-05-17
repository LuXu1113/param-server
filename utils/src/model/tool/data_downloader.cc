#include "model/tool/data_downloader.h"

#include <butil/logging.h>
#include "toolkit/channel.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/fs_agent.h"

using std::string;
using ps::toolkit::Channel;
using ps::toolkit::MPIAgent;
using ps::toolkit::FSAgent;
using ps::toolkit::DataReader;

namespace ps {
namespace model {

DataDownloader::DataDownloader() :
  task_id_(-1),
  tasks_(),
  data_chan_() {
}

void DataDownloader::add_task(const string& path, const string& converter, const string& done_file) {
  LOG(INFO) << "new task: path = " << path << ", converter = " << converter << ", donefile = " << done_file;
  tasks_.push_back({path, converter, done_file});
}

bool DataDownloader::next_task() {
  CHECK(!data_chan_);
  if (task_id_ + 1 >= (int)tasks_.size()) {
    return false;
  }
  ++task_id_;
  return true;
}

void DataDownloader::restart(){
  MPIAgent::mpi_barrier_group();
  task_id_ = -1;
}

const struct DataDownloaderTask& DataDownloader::task_info() {
  CHECK(task_id_ < (int)tasks_.size());
  return tasks_[task_id_];
}

Channel<string> DataDownloader::get_data() {
  CHECK(task_id_ < (int)tasks_.size());

  struct DataDownloaderTask task = tasks_[task_id_];
  LOG(INFO) << "path = " << task.path << ", donefile = " << task.done_file;

  while (true) {
    bool ok = false;
    if (MPIAgent::mpi_rank_group() == 0) {
      ok = (task.done_file.empty() || FSAgent::fs_exists(task.done_file));
    }
    MPIAgent::mpi_bcast_group(&ok, 1, 0);
    if (ok) {
      break;
    }
    fprintf(stdout, "Waiting done file %s\n", tasks_[task_id_].done_file.c_str());
    sleep(60);
  }

  data_chan_ = DataReader::data_read_from(task.path, task.converter);
  while (!(data_chan_->closed())) {
    sleep(1);
  }

  MPIAgent::mpi_barrier_group();
  return std::move(data_chan_);
}

} // namespace model
} // namespace ps

