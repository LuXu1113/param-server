#ifndef UTILS_INCLUDE_TOOLKIT_DATA_READER_H_
#define UTILS_INCLUDE_TOOLKIT_DATA_READER_H_

#include <stdio.h>
#include <stdio_ext.h>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <butil/logging.h>
#include "toolkit/archive.h"
#include "toolkit/channel.h"
#include "toolkit/string_agent.h"
#include "toolkit/thread_group.h"
#include "toolkit/fs_agent.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/parallel_data_processor.h"

namespace ps {
namespace toolkit {

using ReadFromDeserializer = std::function<void (const std::string& path, const std::string& converter, Channel<std::string> out_chan)>;

class DataReader {
 public:
  DataReader() = delete;
  DataReader(const DataReader&) = delete;
  ~DataReader() = delete;

  static void set_default_capacity(size_t capacity);
  static void set_default_block_size(int block_size);
  static void set_default_thread_num(int thread_num);

  static size_t default_capacity();
  static int default_block_size();
  static int default_thread_num();

  static bool read_from_line_deserialize(FILE *fp, std::string& x) {
    thread_local LineFileReader reader;
    if (!reader.getline(fp)) {
      return false;
    }
    x = std::string(reader.get());
    return true;
  }

  static ReadFromDeserializer read_from_line_deserializer() {
    return [](const std::string& path, const std::string& converter, Channel<std::string> out_chan) {
      std::shared_ptr<FILE> file = FSAgent::fs_open(path, "r", converter);
      __fsetlocking(&*file, FSETLOCKING_BYCALLER);
      ChannelWriter<std::string> writer(out_chan);
      std::string x;
      while (writer && read_from_line_deserialize(&*file, x)) {
        writer << std::move(x);
      }
      writer.flush();
    };
  }

  static Channel<std::string> local_read_from(
      const std::vector<std::string>& paths,
      const std::string& converter = "",
      ReadFromDeserializer deserializer = read_from_line_deserializer(),
      int thread_num = default_thread_num()) {

    Channel<std::string> out_chan = make_channel<std::string>();
    out_chan->set_capacity(default_capacity());
    out_chan->set_block_size(default_block_size());

    std::atomic<size_t> *cursor = new std::atomic<size_t>(0);
    out_chan = ParallelDataProcessor::run_parallel<std::string>(out_chan, [paths, converter, deserializer = std::move(deserializer), cursor](Channel<std::string> chan) {
      size_t task;
      for (task = (*cursor)++; task < paths.size(); task = (*cursor)++) {
        deserializer(paths[task], converter, chan);
      }
    }, thread_num);
    delete(cursor);

    return std::move(out_chan);
    // return {&*out_chan, [out_chan](void*) {
    //   out_chan->close();
    // }};
  }

  static Channel<std::string> local_read_from(
      const std::string& path,
      const std::string& converter = "",
      ReadFromDeserializer deserializer = read_from_line_deserializer(),
      int thread_num = default_thread_num()) {
    return local_read_from(FSAgent::fs_list(path), converter, deserializer, thread_num);
  }

  static Channel<std::string> data_read_from(
      const std::string& path,
      const std::string& converter = "",
      ReadFromDeserializer deserializer = read_from_line_deserializer(),
      int thread_num = default_thread_num()) {

    LOG(INFO) << "rank " << MPIAgent::mpi_rank_group() << ", read from " << path;
    MPIAgent::mpi_check_consistency_group(&path, 1);
    std::vector<std::string> all_paths;
    if (MPIAgent::mpi_rank_group() == 0) {
      all_paths = FSAgent::fs_list(path);
    }
    MPIAgent::mpi_bcast_group(&all_paths, 1, 0);

    std::vector<std::string> local_paths;
    for (size_t i = MPIAgent::mpi_rank_group(); i < all_paths.size(); i += MPIAgent::mpi_size_group()) {
      local_paths.push_back(all_paths[i]);
    }

    Channel<std::string> out_chan = local_read_from(local_paths, converter, deserializer, thread_num);

    return std::move(out_chan);
    // return {&*out_chan, [out_chan = std::move(out_chan)](void*) mutable {
    //   CHECK(out_chan.unique());
    //   out_chan = NULL;
    //   MPIAgent::mpi_barrier_group();
    // }};
  }
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_DATA_READER_H_

