#include "toolkit/data_reader.h"

#include "toolkit/string_agent.h"

using std::string;

namespace ps {
namespace toolkit {

static size_t default_capacity_ = 1000000;
static int    default_block_size_ = 8192;
static int    default_thread_num_ = 1;

void DataReader::set_default_capacity(size_t capacity) {
  default_capacity_ = capacity;
}
void DataReader::set_default_block_size(int block_size) {
  default_block_size_ = block_size;
}
void DataReader::set_default_thread_num(int thread_num) {
  default_thread_num_ = thread_num;
}

size_t DataReader::default_capacity() {
  return default_capacity_;
}
int DataReader::default_block_size() {
  return default_block_size_;
}
int DataReader::default_thread_num() {
  return default_thread_num_;
}

} // namespace toolkit
} // namespace ps

