#ifndef UTILS_INCLUDE_MODEL_PLUGIN_COMMON_PLUGIN_H_
#define UTILS_INCLUDE_MODEL_PLUGIN_COMMON_PLUGIN_H_

#include "runtime/config_manager.h"
#include "model/data/thread_local_data.h"

namespace ps {
namespace model {

class Plugin {
 public:
  Plugin() = default;
  Plugin(const Plugin&) = delete;
  ~Plugin() = default;

  virtual void initialize() {
  }
  virtual void finalize() {
  }
  virtual void load_model(const std::string& path) {
  }
  virtual void save_model(const std::string& path) {
  }
  virtual void preprocess(ThreadLocalData *data) {
  }
  virtual void feed_forward(ThreadLocalData *data) {
  }
  virtual void back_propagate(ThreadLocalData *data) {
  }
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_PLUGIN_COMMON_PLUGIN_H_

