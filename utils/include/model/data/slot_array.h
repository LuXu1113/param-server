#ifndef UTILS_INCLUDE_MODEL_DATA_SLOT_ARRAY_H_
#define UTILS_INCLUDE_MODEL_DATA_SLOT_ARRAY_H_

#include <vector>
#include <map>
#include <string>
#include <butil/logging.h>
#include "param_table/data/sparse_kv_ver1.h"

namespace ps {
namespace model {

class SlotArray {
 public:
  explicit SlotArray(const int default_val = -1) : default_val_(default_val) {}

  int default_val() {
    return default_val_;
  }
  void set(int i, const int val) {
    CHECK(i >= 0);
    if (i >= (int)data_.size()) {
      data_.resize(i + 1, default_val_);
    }
    data_[i] = val;
  }
  const int get(int i) const {
    if (i >= 0 && i < (int)data_.size()) {
      return data_[i];
    }
    return default_val_;
  }
  int size() const {
    return (int)data_.size();
  }

 private:
  std::vector<int> data_;
  int default_val_;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DATA_SLOT_ARRAY_H_

