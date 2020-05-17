#ifndef UTILS_INCLUDE_PARAM_TABLE_SPARSE_KV_VER1_TABLE_
#define UTILS_INCLUDE_PARAM_TABLE_SPARSE_KV_VER1_TABLE_

#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "param_table/data/sparse_kv_ver1.h"

namespace ps {
namespace param_table {

class SparseKVVer1Shard {
 public:
  SparseKVVer1Shard();
  SparseKVVer1Shard(const SparseKVVer1Shard&) = delete;
  ~SparseKVVer1Shard();

  int resize(uint64_t size);
  int save(const std::string& file);
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value);
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value);
  int pull(const std::vector<SparseFeatureVer1>& key, std::vector<SparseValueVer1> *value, const bool is_training);
  int time_decay();
  int shrink();
  uint64_t feature_num();

 private:
  absl::flat_hash_map<SparseKeyVer1, SparseValueVer1> data_;
  absl::Mutex rw_mutex_;
};

class SparseKVVer1Table {
 public:
  SparseKVVer1Table();
  SparseKVVer1Table(const std::string &name);
  SparseKVVer1Table(const SparseKVVer1Table&) = delete;
  ~SparseKVVer1Table();

  const std::string& name() const;

  int resize(uint64_t size);
  int save(const std::string& path);
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value);
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value);
  int pull(const std::vector<SparseFeatureVer1>& key, std::vector<SparseValueVer1> *value, const bool is_training);
  int time_decay();
  int shrink();
  uint64_t feature_num();

 private:
  std::string name_;
  std::vector<SparseKVVer1Shard> shard_;
};

class SparseKVVer1TableServer {
 public:
  SparseKVVer1TableServer();
  SparseKVVer1TableServer(const SparseKVVer1TableServer&) = delete;
  ~SparseKVVer1TableServer();

  int create(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int save(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int assign(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int push(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int pull(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int time_decay(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int shrink(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int feature_num(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);

 private:
  absl::flat_hash_map<std::string, SparseKVVer1Table*> tables_;

}; // DenseTable

class SparseKVVer1TableClient {
 public:
  SparseKVVer1TableClient();
  SparseKVVer1TableClient(const SparseKVVer1TableClient&) = delete;
  ~SparseKVVer1TableClient() = default;

  const std::string& name() const;

  int create(const std::string& name);
  int save(const std::string& path) const;
  int assign(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value) const;
  int push(const std::vector<SparseFeatureVer1>& key, const std::vector<SparseValueVer1>& value) const;
  int pull(const std::vector<SparseFeatureVer1>&key, std::vector<SparseValueVer1> *value, const bool is_training) const;
  int time_decay() const;
  int shrink() const;
  uint64_t feature_num() const;

 private:
  std::string name_;

}; // DenseTableClient

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_SPARSE_KV_VER1_TABLE_

