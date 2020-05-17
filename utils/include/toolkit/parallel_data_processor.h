#ifndef UTILS_INCLUDE_TOOLKIT_PARALLEL_DATA_PROCESSOR_H_
#define UTILS_INCLUDE_TOOLKIT_PARALLEL_DATA_PROCESSOR_H_

#include <utility>
#include <functional>
#include <memory>
#include <vector>
#include <butil/logging.h>
#include "toolkit/channel.h"
#include "toolkit/thread_group.h"

namespace ps {
namespace toolkit {

class ParallelDataProcessor {
 public:
  ParallelDataProcessor() = delete;
  ParallelDataProcessor(const ParallelDataProcessor&) = delete;
  ~ParallelDataProcessor() = delete;

  template<class T>
  static Channel<T> run_parallel(Channel<T> chan,
                                 std::function<void (Channel<T>)> func,
                                 int thread_num = parallel_run_num()) {
    CHECK(thread_num >= 1);
    std::shared_ptr<ThreadGroup> thrgrp = std::make_shared<ThreadGroup>(thread_num);
    thrgrp->start([chan, func = std::move(func)](int tid) {
      func(chan);
    });
    thrgrp->join();
    chan->close();
    return chan;
  }

  template<class T>
  static Channel<T> run_parallel(std::function<void (Channel<T>)> func, int thread_num = parallel_run_num()) {
    Channel<T> chan = make_channel<T>();
    return run_parallel(chan, std::move(func), thread_num);
  }

  template<class T, class TT, class FUNC>
  static Channel<T> run_block_map(Channel<TT> in_chan, FUNC&& func, int thread_num = parallel_run_num()) {
    Channel<T> out_chan = make_channel<T>(in_chan);
    return run_parallel<T>(out_chan, [in_chan, func = std::forward<FUNC>(func), thread_num](Channel<T> out_chan) {
      std::vector<TT> input;
      std::vector<T> output;
      while (in_chan->read(input) != 0) {
        output.clear();
        func(input, output);
        out_chan->write(std::move(output));
      }
    }, thread_num);
  }

  template<class T, class TT, class FUNC>
  static Channel<T> run_flat_map(Channel<TT> in_chan, FUNC&& func, int thread_num = parallel_run_num()) {
    return run_block_map<T>(in_chan, [func = std::forward<FUNC>(func)](std::vector<TT>& input, std::vector<T>& output) {
      for (TT& x : input) {
        func(x, output);
      }
    }, thread_num);
  }

  template<class T, class TT, class FUNC>
  static Channel<T> run_map(Channel<TT> in_chan, FUNC&& func, int thread_num = parallel_run_num()) {
    return run_block_map<T>(in_chan, [func = std::forward<FUNC>(func)](std::vector<TT>& input, std::vector<T>& output) {
      for (TT& x : input) {
        output.push_back(func(x));
      }
    }, thread_num);
  }
};

} // namespace toolkit
} // namespace ps
#endif

