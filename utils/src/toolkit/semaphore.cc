#include "toolkit/semaphore.h"

#include <type_traits>
#include <butil/logging.h>

namespace ps {
namespace toolkit {

template<class FUNC, class... ARGS>
static auto ignore_signal_call(FUNC func, ARGS && ... args) -> typename std::result_of<FUNC(ARGS...)>::type {
  for (;;) {
    auto err = func(args...);

    if (err < 0 && errno == EINTR) {
      LOG(INFO) << "Signal is caught. Ignored.";
      continue;
    }

    return err;
  }
}

Semaphore::Semaphore() {
  PCHECK(0 == sem_init(&sem_, 0, 0));
}

Semaphore::~Semaphore() {
  PCHECK(0 == sem_destroy(&sem_));
}

void Semaphore::post() {
  PCHECK(0 == sem_post(&sem_));
}

void Semaphore::wait() {
  PCHECK(0 == ignore_signal_call(sem_wait, &sem_));
}

bool Semaphore::try_wait() {
  int err = 0;
  PCHECK((err = ignore_signal_call(sem_trywait, &sem_), err == 0 || errno == EAGAIN));
  return err == 0;
}

} // namespace toolkit
} // namespace ps

