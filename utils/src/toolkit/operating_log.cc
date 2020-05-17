#include "toolkit/operating_log.h"
#include <butil/logging.h>

namespace ps {
namespace toolkit {

OperatingLog::OperatingLog() :
  count_(0),
  total_(),
  max_(),
  min_(),
  mtx_() {
  clear();
}

OperatingLog::~OperatingLog() {
}

void OperatingLog::set_name(const std::string& name) {
  name_ = name;
}

void OperatingLog::record(const absl::Time& begin, const absl::Time& end) {
  absl::Duration duration = end - begin;
  mtx_.WriterLock();
  ++count_;
  total_ += duration;

  if (max_ < duration) {
    max_ = duration;
  }
  if (min_ > duration) {
    min_ = duration;
  }
  mtx_.WriterUnlock();
}

void OperatingLog::clear() {
  mtx_.WriterLock();
  count_ = 0;
  total_ = absl::Nanoseconds(0);
  max_   = absl::Nanoseconds(0);
  min_   = absl::Nanoseconds(0x7FFFFFFFFFFFFFFFLL);
  mtx_.WriterUnlock();
}

void OperatingLog::log() {
  mtx_.ReaderLock();
  LOG(INFO) << name_ << ", count: " << count_
            << ", total: " << total_ << ", avg: " << total_ / (double)count_
            << ", max: " << max_ << ", min: " << min_;
  mtx_.ReaderUnlock();
}

} // namespace toolkit
} // namespace ps

