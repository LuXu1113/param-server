#include "toolkit/string_agent.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <string>

using std::string;

namespace ps {
namespace toolkit {

size_t StringAgent::count_if(const char* s, const char c) {
  size_t count = 0;
  for (const char *iter = s; *iter != 0 && c == *iter; ++iter) {
    ++count;
  }
  return count;
}

size_t StringAgent::count_if_not(const char* s, const char c) {
  size_t count = 0;
  for (const char *iter = s; *iter != 0 && c != *iter; ++iter) {
    ++count;
  }
  return count;
}

size_t StringAgent::count_spaces(const char *s) {
  size_t count = 0;
  for (const char *iter = s; *iter != 0 && isspace(*iter); ++iter) {
    ++count;
  }
  return count;
}

size_t StringAgent::count_nonspaces(const char *s) {
  size_t count = 0;
  for (const char *iter = s; *iter != 0 && !isspace(*iter); ++iter) {
    ++count;
  }
  return count;
}

LineFileReader::LineFileReader() :
  buffer_(NULL),
  buf_size_(0),
  length_(0) {
}

LineFileReader::~LineFileReader() {
  if (NULL != buffer_) {
    free(buffer_);
  }
}

char *LineFileReader::getline(FILE *f) {
  return this->getdelim(f, '\n');
}

char *LineFileReader::getdelim(FILE *f, char delim) {
  ssize_t ret = ::getdelim(&buffer_, &buf_size_, delim, f);

  if (ret >= 0) {
    if (ret >= 1 && buffer_[ret - 1] == delim) {
      buffer_[--ret] = 0;
    }

    length_ = (size_t)ret;
    return buffer_;
  } else {
    length_ = 0;
    CHECK(feof(f));
    return NULL;
  }
}

char *LineFileReader::get() {
  return buffer_;
}

size_t LineFileReader::length() {
  return length_;
}

} // namespace toolkit
} // namespace ps

