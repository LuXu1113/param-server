#include "toolkit/managed_thread.h"

#include <butil/logging.h>

using std::thread;
using std::function;

namespace ps {
namespace toolkit {

ManagedThread::ManagedThread() :
  is_active_(false),
  is_terminate_(false) {
  thr_ = thread([this]() {
    run_thread();
  });
}

ManagedThread::~ManagedThread() {
  CHECK(!is_active_);
  is_terminate_ = true;
  sem_start_.post();
  thr_.join();
}

bool ManagedThread::is_active() const {
  return is_active_;
}

void ManagedThread::start(function<void()>&& func) {
  CHECK(!is_active_);
  is_active_ = true;
  func_ = std::move(func);
  sem_start_.post();
}

void ManagedThread::join() {
  CHECK(is_active_);
  sem_finish_.wait();
  is_active_ = false;
}

void ManagedThread::run_thread() {
  while (!is_terminate_) {
    sem_start_.wait();

    if (is_terminate_) {
      break;
    };

    func_();
    sem_finish_.post();
  }
}

} // namespace toolkit
} // namespace ps

