#ifndef UTILS_INCLUDE_TOOLKIT_OPERATING_LOG_H_
#define UTILS_INCLUDE_TOOLKIT_OPERATING_LOG_H_

#include <string>
#include "absl/time/time.h"
#include "absl/synchronization/mutex.h"

namespace ps {
namespace toolkit {

class OperatingLog {
 public:
  OperatingLog();
  ~OperatingLog();

  void set_name(const std::string& name);
  void record(const absl::Time& begin, const absl::Time& end);
  void clear();
  void log();

 private:
  std::string name_;
  uint64_t count_;
  absl::Duration total_;
  absl::Duration max_;
  absl::Duration min_;
  uint64_t begin_timestamp_;

  absl::Mutex mtx_;
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_OPERATING_LOG_H_

