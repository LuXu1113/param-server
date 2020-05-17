#ifndef UTILS_INCLUDE_TOOLKIT_MANAGED_THREAD_H_
#define UTILS_INCLUDE_TOOLKIT_MANAGED_THREAD_H_

#include <functional>
#include <thread>
#include "toolkit/semaphore.h"

namespace ps {
namespace toolkit {

class alignas(64) ManagedThread {
 public:
  ManagedThread();
  ~ManagedThread();

  bool is_active() const;
  void start(std::function<void()> && func);
  void join();

 private:
  bool is_active_;
  bool is_terminate_;
  std::function<void()> func_;
  Semaphore sem_start_;
  Semaphore sem_finish_;
  std::thread thr_;

  void run_thread();
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_MANAGED_THREAD_H_

