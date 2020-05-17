#ifndef UTILS_INCLUDE_MODEL_TOOL_DATA_DOWNLOADER_H_
#define UTILS_INCLUDE_MODEL_TOOL_DATA_DOWNLOADER_H_

#include <string>
#include "toolkit/channel.h"
#include "toolkit/data_reader.h"

namespace ps {
namespace model {

struct DataDownloaderTask {
  std::string path;
  std::string converter;
  std::string done_file;
};

class DataDownloader {
 public:
  DataDownloader();
  DataDownloader(const DataDownloader&) = delete;
  ~DataDownloader() = default;

  void add_task(const std::string& path, const std::string& converter, const std::string& done_file);
  bool next_task();
  void restart();
  const struct DataDownloaderTask& task_info();

  ps::toolkit::Channel<std::string> get_data();

 private:
  int task_id_;
  std::vector<DataDownloaderTask> tasks_;
  ps::toolkit::Channel<std::string> data_chan_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_TOOL_DATA_DOWNLOADER_H_

