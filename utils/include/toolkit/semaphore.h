#ifndef UTILS_INCLUDE_TOOLKIT_SEMAPHORE_H_
#define UTILS_INCLUDE_TOOLKIT_SEMAPHORE_H_

#include <semaphore.h>

namespace ps {
namespace toolkit {

class Semaphore {
 public:
  Semaphore();
  Semaphore(const Semaphore&) = delete;
  ~Semaphore();

  void post();
  void wait();
  bool try_wait();

 private:
   sem_t sem_;
};

// #include <mutex>
// #include <condition_variable>
//
// class Semaphore {
//  public:
//   Semaphore() : count(1) {}
//   explicit Semaphore(int value) : count(value) {}
//   void down() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (--count < 0) {
//       cv.wait(lck);
//     }
//   }
//   void up() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (++count <= 0) {
//       cv.notify_one();
//     }
//   }
//  private:
//   int count;
//   std::mutex mtk;
//   std::condition_variable cv;
// };

} // namespace toolkit
} // namespace ps

