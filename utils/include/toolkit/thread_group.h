#ifndef UTILS_INCLUDE_TOOLKIT_THREAD_GROUP_H_
#define UTILS_INCLUDE_TOOLKIT_THREAD_GROUP_H_

#include <pthread.h>
#include <atomic>
#include <functional>
#include <vector>
#include "toolkit/managed_thread.h"

namespace ps {
namespace toolkit {

class ScopeExit {
 public:
  explicit ScopeExit(std::function<void ()> f) : f_(std::move(f)) {
  }
  ScopeExit(const ScopeExit&) = delete;
  ~ScopeExit() {
    f_();
  }

 private:
  std::function<void ()> f_;
};

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count = 1);
  ThreadBarrier(const ThreadBarrier&) = delete;
  ~ThreadBarrier();
  void reset(int count);
  void wait();
 private:
  pthread_barrier_t barrier_;
};

class ThreadGroup {
 public:
  explicit ThreadGroup(int thread_num = 0);
  ThreadGroup(const ThreadGroup&) = delete;
  ~ThreadGroup();

  int real_thread_num();
  int parallel_num();

  void set_real_thread_num(int thread_num);
  void set_parallel_num(int parallel_num);
  bool joinable();

  void run(std::function<void (int)> func);
  void start(std::function<void (int)> func);

  void join();
  static int& thread_id();
  static ThreadGroup*& parent_group();

  void barrier_wait();

 private:
  int parallel_num_ = 1;
  std::vector<ManagedThread> threads_;
  ThreadBarrier barrier_;
  std::function<void (int)> func_;
};

int parallel_run_id();

ThreadGroup& local_thread_group();

ThreadGroup& global_write_thread_group();

int parallel_run_num(ThreadGroup& thrgrp = local_thread_group());

void parallel_run_barrier_wait();

void parallel_run_barrier_wait(ThreadGroup& thrgrp);

template<class THREAD_FUNC>
void parallel_run(THREAD_FUNC&& func, ThreadGroup& thrgrp = local_thread_group()) {
  thrgrp.run([&func](int i) {
    func(i);
  });
}

template<class THREAD_FUNC>
void parallel_run_range(uint64_t n, THREAD_FUNC&& func, ThreadGroup& thrgrp = local_thread_group()) {
  int thr_num = thrgrp.parallel_num();
  thrgrp.run([n, &func, thr_num](int i) {
    func(i, n * i / thr_num, n * (i + 1) / thr_num);
  });
}

template<class THREAD_FUNC>
void parallel_run_dynamic(int n, THREAD_FUNC&& func, ThreadGroup& thrgrp = local_thread_group()) {
  std::atomic<int> counter(0);
  thrgrp.run([n, &counter, &func](int thr_id) {
    int i;
    while (i = counter++, i < n) {
      func(thr_id, i);
    }
  });
}

} // namespace toolkit
} // namespace ps


#endif // UTILS_INCLUDE_TOOLKIT_THREAD_GROUP_H_

