#ifndef UTILS_INCLUDE_MODEL_PLUGIN_FM_PLUGIN_H_
#define UTILS_INCLUDE_MODEL_PLUGIN_FM_PLUGIN_H_

#include <math.h>
#include <vector>
#include "runtime/config_manager.h"
#include "param_table/data/sparse_kv_ver1.h"
#include "model/data/instance.h"
#include "model/data/thread_local_data.h"
#include "model/plugin/common/plugin.h"

namespace ps {
namespace model {

class FMPlugin final : public Plugin {
 public:
  void initialize() override;
  void finalize() override {}
  void load_model(const std::string& path) override {}
  void save_model(const std::string& path) override {}
  void preprocess(ThreadLocalData *data) override {}
  void feed_forward(ThreadLocalData *data) override;
  void back_propagate(ThreadLocalData *data) override;

 private:
  int dim_;
  std::vector<int> slots_;
  float create_threshold_;
  bool joint_train_ = false;
};

} // namespace model
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_PLUGIN_FM_PLUGIN_H_

