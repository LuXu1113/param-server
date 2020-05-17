#ifndef UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_EMBEDDING_VER1_H_
#define UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_EMBEDDING_VER1_H_

#include <string>
#include "runtime/config_manager.h"
#include "param_table/data/sparse_kv_ver1.h"

namespace ps {
namespace param_table {

struct SparseEmbeddingVer1 {
  SparseSlotVer1 slot_;
  int silent_days_;

  float count_;
  float ada_d2sum_;

  std::vector<float> embedding_;
  std::vector<float> ada_g2sum_;

  uint64_t version_;
  float delta_score_;
};

SparseEmbeddingVer1 sparse_embedding_ver1_default();
int sparse_embedding_ver1_init(SparseEmbeddingVer1 *value, const ps::runtime::TrainingRule& rule);
int sparse_embedding_ver1_push(SparseEmbeddingVer1 *value, const SparseEmbeddingVer1& grad, const ps::runtime::TrainingRule& rule);
int sparse_embedding_ver1_merge(SparseEmbeddingVer1 *value, const SparseEmbeddingVer1& new_value, const ps::runtime::TrainingRule& rule);
int sparse_embedding_ver1_to_string(const SparseKeyVer1& key, const SparseEmbeddingVer1& value, std::string *str);
int sparse_embedding_ver1_time_decay(SparseEmbeddingVer1 *value, const ps::runtime::TrainingRule& rule);
bool sparse_embedding_ver1_shrink(const SparseEmbeddingVer1& value, const ps::runtime::TrainingRule& rule);

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_DATA_SPARSE_EMBEDDING_VER1_H_

