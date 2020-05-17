#include "model/data/record.h"

#include <iostream>
#include <butil/logging.h>
#include "toolkit/string_agent.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using ps::toolkit::Channel;
using ps::toolkit::StringAgent;
using ps::param_table::SparseFeatureVer1;

namespace ps {
namespace model {

static inline bool operator<(const SparseFeatureVer1& a, const SparseFeatureVer1& b) {
  return (a.sign_ < b.sign_) || ((a.sign_ == b.sign_) && (a.slot_ < b.slot_));
}

static inline bool operator==(const SparseFeatureVer1& a, const SparseFeatureVer1& b) {
  return (a.sign_ == b.sign_) && (a.slot_ == b.slot_);
}

inline void parse_sign_slot(const char **line, Record *rec) {
  uint64_t sign;
  uint32_t slot;
  char *cursor;

  sign = (uint64_t)strtoull((*line), &cursor, 10);
  if (cursor == (*line)) {
    LOG(ERROR) << "sign: " << sign << ", line: " << (*line);
    rec->show_ = -1;
    return;
  }
  (*line) = cursor;

  if ((**line) != ':') {
    LOG(ERROR) << "line: " << (*line);
    rec->show_ = -1;
    return;
  }
  ++(*line);

  if (isspace(**line)) {
    LOG(ERROR) << "line: " << (*line);
    rec->show_ = -1;
    return;
  }

  slot = (uint32_t)strtoul((*line), &cursor, 10);
  if (cursor == (*line)) {
    LOG(ERROR) << "slot: " << slot << ", line: " << (*line);
    rec->show_ = -1;
    return;
  }
  (*line) = cursor;

  rec->feas_.push_back({sign, slot});
}

void parse_record_withoutlineid(const char *line, Record& rec) {
}

void parse_record(const char *input, Record& rec) {
  const char *line = input;
  char *cursor;

  // leading spaces
  line += StringAgent::count_spaces(line);

  // line ID
  rec.lineid_.assign(line, StringAgent::count_nonspaces(line));
  line += rec.lineid_.length();

  // show
  rec.show_ = (int)strtol(line, &cursor, 10);
  if (cursor == line || rec.show_ < 1) {
    LOG(ERROR) << "show: " << rec.show_ << ", input: " << input;
    rec.show_ = -1;
    return;
  }
  line = cursor;

  // click
  rec.clk_ = (int)strtol(line, &cursor, 10);
  if (cursor == line || rec.clk_ < 0) {
    LOG(ERROR) << "click: " << rec.clk_ << ", input: " << input;
    rec.show_ = -1;
    return;
  }
  line = cursor;

  // check
  if (rec.clk_ > rec.show_) {
    LOG(ERROR) << "show: " << rec.show_ << ", click: " << rec.clk_ << ", input: " << input;
    rec.show_ = -1;
    return;
  }

  rec.feas_.clear();
  rec.vec_feas_.clear();
  rec.addition_ins_.clear();

  thread_local vector<std::vector<SparseFeatureVer1> > vv_feas;
  thread_local vector<SparseFeatureVer1> v_feas;
  thread_local std::vector<float> vf;

  const char *line_end = line + std::strlen(line);

  rec.bid_ = 1.0;
  while (*(line += StringAgent::count_spaces(line)) != '\0') {
    if ((*line) == '@') { // bid
      line++;
      size_t len = std::find_if_not(line, line_end, [](char c) { return std::isalnum(c) != 0 || c == '_';}) - line;
      if (len <= 0 || *(line + len) != ':') {
        rec.show_ = -1;
        LOG(ERROR) << "Wrong line: " << input;
        cout << "[FUNCTION:" << __func__ << ", LINE:" << __LINE__ << "] Wrong line:"  << input << endl;
        return;
      }
      std::string name(line, len);
      line += len ;
      if (*line == ':') {
        float bid = (float)strtof(line + 1, &cursor);
        if (bid == 0.0) {
          LOG(WARNING) << "Wrong line: " << input << ", bid value equals 0.0" << line;
          bid = 1.0;
        }
        rec.bid_ = bid;
        line = cursor;
      }
    } else if ((*line) == '$') {
      line++;
      size_t len = std::find_if_not(line, line_end,[](char c) { return std::isalnum(c) != 0 || c == '_';}) - line;
      if (len <=0 || *(line + len) != '|' ) {
        rec.show_ = -1;
        std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << input << std::endl;
        return;
      }
      std::string name(line, len);
      vv_feas.clear();
      line += len;
      while (*line == '|') {
        line++;
        v_feas.clear();
        while (true) {
          uint64_t sign;
          int slot;
          sign = (uint64_t)strtoull(line, &cursor, 10);
          if (cursor == line) {
            break;
          }
          line = cursor;
          if(*line != ':') {
            rec.show_ = -1;
            std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << input << std::endl;
            return;
          }

          line++;
          slot = (int)strtol(line, &cursor, 10);
          if(cursor == line){
            rec.show_ = -1;
            std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << input << std::endl;
            return;
          }
          if (slot != slot) {
            rec.show_ = -1;
            std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << input << std::endl;
            return;
          }
          v_feas.push_back({sign, (unsigned int)slot});
          line = cursor;
          if(!(line == line_end || *line == ',' || *line == '|' || *line == ' ' || *line == '\n' || *line == '\t')){
            rec.show_ = -1;
            std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << input << std::endl;
            return;
          }
          if (line == line_end || *line == '|' || *line == ' ' || *line == '\n' || *line == '\t') break;
          line++;
        }
        vv_feas.push_back(v_feas);
      }
      CHECK(rec.addition_ins_.insert({name, vv_feas}).second);
    } else if (*line == '#') {
      line++;
      size_t len = std::find_if_not(line, line_end, [](char c) { return std::isalnum(c) != 0 || c == '_';}) - line;
      if (len <= 0 || *(line + len) != ':') {
        rec.show_ = -1;
        LOG(ERROR) << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line;
        std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line << std::endl;
        return;
      }
      std::string name(line, len);
      line += len ;
      vf.clear();
      while (*line == ':') {
        float val;
        val = strtof(line + 1, &cursor);
        if (cursor <= line) {
          rec.show_ = -1;
          LOG(ERROR) << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line;
          std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line << std::endl;
          return;
        }
        vf.push_back(val);
        line = cursor;
      }
      if (!rec.vec_feas_.insert({name, vf}).second) {
        rec.show_ = -1;
        LOG(ERROR) << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line;
        std::cout << "[FUNCTION:" << __func__ << ",LINE:" <<__LINE__<< "]Wrong line:"  << line << std::endl;
        return;
      }
    } else {
      parse_sign_slot(&line, &rec);
    }
  }
}

vector<Record> parse_records_withoutlineid(Channel<string> channel) {
  vector<Record> res;
  return std::move(res);
}
vector<Record> parse_records(Channel<string> channel) {
  vector<Record> res;
  return std::move(res);
}

} // namespace model
} // namespace ps

