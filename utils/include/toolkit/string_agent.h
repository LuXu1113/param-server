#ifndef UTILS_INCLUDE_TOOLKIT_STRING_AGENT_H_
#define UTILS_INCLUDE_TOOLKIT_STRING_AGENT_H_

#include <string>
#include <butil/logging.h>

namespace ps {
namespace toolkit {

class StringAgent {
 public:
  StringAgent() = delete;
  static size_t count_spaces(const char *s);
  static size_t count_nonspaces(const char* s);
  static size_t count_if(const char* s, char c);
  static size_t count_if_not(const char* s, char c);
};

// A helper class for reading lines from file. A line buffer is maintained. It doesn't need to know the maximum possible length of a line.
class LineFileReader {
 public:
  LineFileReader();
  LineFileReader(const LineFileReader&) = delete;
  ~LineFileReader();

  char *getline(FILE* f);
  char *getdelim(FILE* f, char delim);
  char *get();
  size_t length();

 private:
  char *buffer_;
  size_t buf_size_;
  size_t length_;
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_STRING_AGENT_H_

