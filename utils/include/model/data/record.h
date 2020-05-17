#ifndef UtILS_INCLUDE_MODEL_DATA_RECORD_H_
#define UtILS_INCLUDE_MODEL_DATA_RECORD_H_

#include <vector>
#include <string>
#include <map>
#include "toolkit/channel.h"
#include "param_table/data/sparse_kv_ver1.h"

namespace ps {
namespace model {

struct Record {
  std::string lineid_;
  int show_;
  int clk_;
  float bid_;
  std::vector<ps::param_table::SparseFeatureVer1> feas_;
  std::map<std::string, std::vector<float> > vec_feas_;
  std::map<std::string, std::vector<std::vector<ps::param_table::SparseFeatureVer1> > > addition_ins_;
};

void parse_record_withoutlineid(const char *line, Record& rec);
void parse_record(const char *input, Record& rec);

std::vector<Record> parse_records_withoutlineid(ps::toolkit::Channel<std::string> channel);
std::vector<Record> parse_records(ps::toolkit::Channel<std::string> channel);

} // namespace model
} // namespace ps

#endif // UtILS_INCLUDE_MODEL_DATA_RECORD_H_


