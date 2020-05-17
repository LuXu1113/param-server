#include "toolkit/thread_group.h"

#include <utility>
#include <vector>
#include <butil/logging.h>

using std::move;
using std::vector;
using std::function;

namespace ps {
namespace toolkit {

ThreadBarrier::ThreadBarrier(int count) {
  CHECK(count >= 1);
  PCHECK(0 == pthread_barrier_init(&barrier_, NULL, count));
}

ThreadBarrier::~ThreadBarrier() {
  PCHECK(0 == pthread_barrier_destroy(&barrier_));
}

void ThreadBarrier::reset(int count) {
  CHECK(count >= 1);
  PCHECK(0 == pthread_barrier_destroy(&barrier_));
  PCHECK(0 == pthread_barrier_init(&barrier_, NULL, count));
}

void ThreadBarrier::wait() {
  int err = pthread_barrier_wait(&barrier_);
  PCHECK((err = pthread_barrier_wait(&barrier_), err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD));
}

ThreadGroup::ThreadGroup(int thread_num) {
  set_real_thread_num(thread_num);
}

ThreadGroup::~ThreadGroup() {
  set_real_thread_num(0);
}

int ThreadGroup::real_thread_num() {
  return (int)threads_.size();
}

int ThreadGroup::parallel_num() {
  return parallel_num_;
}

void ThreadGroup::set_real_thread_num(int thread_num) {
  CHECK(thread_num >= 0);
  CHECK(!joinable());

  if (thread_num == (int)threads_.size()) {
    return;
  }

  threads_ = vector<ManagedThread>(thread_num);
  parallel_num_ = (thread_num == 0) ? 1 : thread_num;
  barrier_.reset(parallel_num_);
}

void ThreadGroup::set_parallel_num(int parallel_num) {
  CHECK(parallel_num >= 1);
  set_real_thread_num(parallel_num);
}

bool ThreadGroup::joinable() {
  return (bool)func_;
}

void ThreadGroup::run(function<void (int)> func) {
  start(move(func));
  join();
}

void ThreadGroup::start(function<void (int)> func) {
  CHECK((bool)func);
  CHECK(!joinable());

  if (threads_.empty()) {
    ScopeExit on_exit([old_id = thread_id(), old_grp = parent_group()]() {
        thread_id() = old_id;
        parent_group() = old_grp;
    });

    thread_id() = 0;
    parent_group() = this;
    func(0);
    return;
  }

  func_ = move(func);
  for (int i = 0; i < parallel_num_; ++i) {
    threads_[i].start([this, i]() {
      thread_id() = i;
      parent_group() = this;
      func_(i);
    });
  }
}

void ThreadGroup::join() {
  CHECK(joinable());
  for (int i = 0; i < (int)threads_.size(); ++i) {
    threads_[i].join();
  }
  func_ = nullptr;
}

int& ThreadGroup::thread_id() {
  thread_local int x = 0;
  return x;
}

ThreadGroup*& ThreadGroup::parent_group() {
  thread_local ThreadGroup *x = NULL;
  return x;
}

void ThreadGroup::barrier_wait() {
  barrier_.wait();
}

int parallel_run_id() {
  return ThreadGroup::thread_id();
}

ThreadGroup& local_thread_group() {
  thread_local ThreadGroup g;
  return g;
}

ThreadGroup& global_write_thread_group() {
  static ThreadGroup g;
  return g;
}

int parallel_run_num(ThreadGroup& thrgrp) {
  return thrgrp.parallel_num();
}

void parallel_run_barrier_wait() {
  ThreadGroup* thrgrp = ThreadGroup::parent_group();
  CHECK(thrgrp != NULL);
  thrgrp->barrier_wait();
}

void parallel_run_barrier_wait(ThreadGroup& thrgrp) {
  thrgrp.barrier_wait();
}

} // namespace toolkit
} // namespace ps

