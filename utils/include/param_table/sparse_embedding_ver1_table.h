#ifndef UTILS_INCLUDE_PARAM_TABLE_SPARSE_EMBEDDING_VER1_TABLE_
#define UTILS_INCLUDE_PARAM_TABLE_SPARSE_EMBEDDING_VER1_TABLE_

#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "param_table/data/sparse_embedding_ver1.h"

namespace ps {
namespace param_table {

class SparseEmbeddingVer1Shard {
 public:
  SparseEmbeddingVer1Shard();
  SparseEmbeddingVer1Shard(const SparseEmbeddingVer1Shard&) = delete;
  ~SparseEmbeddingVer1Shard();

  int resize(uint64_t size);
  int save(const std::string& file);
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value);
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value);
  int pull(const std::vector<SparseFeatureVer1>& key, std::vector<SparseEmbeddingVer1> *value, const bool is_training);
  int time_decay();
  int shrink();
  uint64_t feature_num();

 private:
  absl::flat_hash_map<SparseKeyVer1, SparseEmbeddingVer1> data_;
  absl::Mutex rw_mutex_;
};

class SparseEmbeddingVer1Table {
 public:
  SparseEmbeddingVer1Table();
  SparseEmbeddingVer1Table(const std::string &name);
  SparseEmbeddingVer1Table(const SparseEmbeddingVer1Table&) = delete;
  ~SparseEmbeddingVer1Table();

  const std::string& name() const;

  int resize(uint64_t size);
  int save(const std::string& path);
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value);
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value);
  int pull(const std::vector<SparseFeatureVer1>& key, std::vector<SparseEmbeddingVer1> *value, const bool is_training);
  int time_decay();
  int shrink();
  uint64_t feature_num();

 private:
  std::string name_;
  std::vector<SparseEmbeddingVer1Shard> shard_;
};

class SparseEmbeddingVer1TableServer {
 public:
  SparseEmbeddingVer1TableServer();
  SparseEmbeddingVer1TableServer(const SparseEmbeddingVer1TableServer&) = delete;
  ~SparseEmbeddingVer1TableServer();

  int create(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int save(const ps::ParamServerRequest& request, ps::ParamServerResponse *response) const;
  int assign(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int push(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int pull(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int time_decay(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int shrink(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int feature_num(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);

 private:
  absl::flat_hash_map<std::string, SparseEmbeddingVer1Table*> tables_;

}; // DenseTable

class SparseEmbeddingVer1TableClient {
 public:
  SparseEmbeddingVer1TableClient();
  SparseEmbeddingVer1TableClient(const SparseEmbeddingVer1TableClient&) = delete;
  ~SparseEmbeddingVer1TableClient() = default;

  const std::string& name() const;

  int create(const std::string& name);
  int save(const std::string& path) const;
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value) const;
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseEmbeddingVer1>& value) const;
  int pull(const std::vector<SparseFeatureVer1>&key, std::vector<SparseEmbeddingVer1> *value, const bool is_training) const;
  int time_decay() const;
  int shrink() const;
  uint64_t feature_num() const;

 private:
  std::string name_;

}; // DenseTableClient

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_SPARSE_EMBEDDING_VER1_TABLE_

