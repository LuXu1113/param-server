#ifndef UTILS_INCLUDE_PARAM_TABLE_SUMMARY_VALUE_VER1_TABLE_
#define UTILS_INCLUDE_PARAM_TABLE_SUMMARY_VALUE_VER1_TABLE_

#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "param_table/data/summary_value_ver1.h"

namespace ps {
namespace param_table {

class SummaryValueVer1Shard {
 public:
  SummaryValueVer1Shard();
  SummaryValueVer1Shard(const SummaryValueVer1Shard&) = delete;
  ~SummaryValueVer1Shard();

  uint64_t mem_size();
  uint64_t size();

  int resize(uint64_t begin, uint64_t end);
  int assign(const std::vector<SummaryValueVer1>&value);
  int push(const std::vector<SummaryValueVer1>&value);
  int pull(std::vector<SummaryValueVer1> *value);

 private:
  std::vector<SummaryValueVer1> data_;
  uint64_t begin_;
  uint64_t end_;
  absl::Mutex rw_mutex_;
};

class SummaryValueVer1Table {
 public:
  SummaryValueVer1Table();
  SummaryValueVer1Table(const std::string &name);
  SummaryValueVer1Table(const SummaryValueVer1Table&) = delete;
  ~SummaryValueVer1Table();

  const std::string& name() const;
  uint64_t size() const;
  uint64_t mem_size();

  int resize(uint64_t size);
  int save(const std::string& path);
  int assign(const std::vector<SummaryValueVer1>& value);
  int push(const std::vector<SummaryValueVer1>& value);
  int pull(std::vector<SummaryValueVer1> *value);

 private:
  std::string name_;
  uint64_t size_;
  std::vector<SummaryValueVer1Shard> shard_;
};

class SummaryValueVer1TableServer {
 public:
  SummaryValueVer1TableServer();
  SummaryValueVer1TableServer(const SummaryValueVer1TableServer&) = delete;
  ~SummaryValueVer1TableServer();

  uint64_t mem_size();

  int create(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int save(const ps::ParamServerRequest& request, ps::ParamServerResponse *response) const;
  int resize(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int assign(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int push(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);
  int pull(const ps::ParamServerRequest& request, ps::ParamServerResponse *response);

 private:
  absl::flat_hash_map<std::string, SummaryValueVer1Table*> tables_;

}; // DenseTable

class SummaryValueVer1TableClient {
 public:
  SummaryValueVer1TableClient();
  SummaryValueVer1TableClient(const SummaryValueVer1TableClient&) = delete;
  ~SummaryValueVer1TableClient() = default;

  const std::string& name() const;
  const uint64_t size() const;

  int create(const std::string& name);
  int resize(const uint64_t size);
  int save(const std::string& path) const;
  int assign(const std::vector<SummaryValueVer1>& value) const;
  int push(const std::vector<SummaryValueVer1>& value) const;
  int pull(std::vector<SummaryValueVer1> *value) const;

 private:
  std::string name_;
  uint64_t size_;
  std::vector<uint64_t> boundaries_;

}; // DenseTableClient

} // namespace param_table
} // namespace ps

#endif // UTILS_INCLUDE_PARAM_TABLE_SUMMARY_VALUE_VER1_TABLE_

