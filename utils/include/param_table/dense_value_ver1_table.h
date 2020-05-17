#ifndef UTILS_INCLUDE_PARAM_TABLE_DENSE_VALUE_VER1_TABLE_
#define UTILS_INCLUDE_PARAM_TABLE_DENSE_VALUE_VER1_TABLE_

#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "param_table/data/dense_value_ver1.h"

namespace ps {
namespace param_table {

class DenseValueVer1Shard {
 public:
  DenseValueVer1Shard();
  DenseValueVer1Shard(const DenseValueVer1Shard&) = delete;
  ~DenseValueVer1Shard();

  uint64_t mem_size();
  uint64_t size();

  int resize(uint64_t begin, uint64_t end);
  int assign(const std::vector<DenseValueVer1>&value);
  int push(const std::vector<DenseValueVer1Push>&value);
  int pull(std::vector<DenseValueVer1Pull> *value);

 private:
  std::vector<DenseValueVer1> data_;
  uint64_t begin_;
  uint64_t end_;
  absl::Mutex rw_mutex_;
};

class DenseValueVer1Table {
 public:
  DenseValueVer1Table();
  DenseValueVer1Table(const std::string &name);
  DenseValueVer1Table(const DenseValueVer1Table&) = delete;
  ~DenseValueVer1Table();

  const std::string& name() const;
  uint64_t size() const;
  uint64_t mem_size();

  int resize(uint64_t size);
  int save(const std::string& path);
  int assign(const std::vector<DenseValueVer1>& value);
  int push(const std::vector<DenseValueVer1Push>& value);
  int pull(std::vector<DenseValueVer1Pull> *value);

 private:
  std::string name_;
  uint64_t size_;
  std::vector<DenseValueVer1Shard> shard_;
};

class DenseValueVer1TableServer {
 public:
  DenseValueVer1TableServer();
  DenseValueVer1TableServer(const DenseValueVer1TableServer&) = delete;
  ~DenseValueVer1TableServer();

  uint64_t mem_size();

  int create(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int save(const ps::ParamServerRequest& request, ps::ParamServerResponse *response) const;
  int resize(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int assign(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int push(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int pull(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);

 private:
  absl::flat_hash_map<std::string, DenseValueVer1Table*> tables_;

}; // DenseTable

class DenseValueVer1TableClient {
 public:
  DenseValueVer1TableClient();
  DenseValueVer1TableClient(const DenseValueVer1TableClient&) = delete;
  ~DenseValueVer1TableClient() = default;

  const std::string& name() const;
  const uint64_t size() const;

  int create(const std::string& name);
  int resize(const uint64_t size);
  int save(const std::string& path) const;
  int assign(const std::vector<DenseValueVer1>& value) const;
  int push(const std::vector<DenseValueVer1Push>& value) const;
  int pull(std::vector<DenseValueVer1Pull> *value) const;

 private:
  std::string name_;
  uint64_t size_;
  std::vector<uint64_t> boundaries_;

}; // DenseTableClient

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_DENSE_VALUE_VER1_TABLE_

