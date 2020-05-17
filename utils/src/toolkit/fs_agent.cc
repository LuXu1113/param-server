#include "toolkit/fs_agent.h"

#include <sys/stat.h>
#include <string.h>
#include <butil/logging.h>
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "toolkit/string_agent.h"
#include "toolkit/shell_agent.h"

using std::vector;
using std::string;
using std::shared_ptr;
using ps::toolkit::LineFileReader;
using ps::toolkit::ShellAgent;

namespace ps {
namespace toolkit {

/**************************************  local file system  **************************************/
static size_t localfs_buffer_size_ = 0;

size_t FSAgent::localfs_buffer_size() {
  return localfs_buffer_size_;
}

void FSAgent::localfs_set_buffer_size(size_t x) {
  localfs_buffer_size_ = x;
}

shared_ptr<FILE> FSAgent::localfs_open_read(const string& path, const string& converter) {
  string cmd = path;
  bool is_pipe = false;

  if (fs_end_with_internal(cmd, ".gz")) {
    fs_add_read_converter_internal(&cmd, &is_pipe, "zcat");
  }
  fs_add_read_converter_internal(&cmd, &is_pipe, converter);

  return fs_open_internal(cmd, is_pipe, "r", localfs_buffer_size_);
}

shared_ptr<FILE> FSAgent::localfs_open_write(const string& path, const string& converter) {
  ShellAgent::shell_execute(absl::StrFormat("mkdir -p $(dirname \"%s\")", path.c_str()));

  string cmd = path;
  bool is_pipe = false;
  if (fs_end_with_internal(cmd, ".gz")) {
    fs_add_write_converter_internal(&cmd, &is_pipe, "gzip");
  }
  fs_add_write_converter_internal(&cmd, &is_pipe, converter);

  return fs_open_internal(cmd, is_pipe, "w", localfs_buffer_size_);
}

int64_t FSAgent::localfs_file_size(const string& path) {
  struct stat buf;
  CHECK(0 == stat(path.c_str(), &buf));
  return (int64_t)buf.st_size;
}

void FSAgent::localfs_remove(const string& path) {
  if (path == "") {
    return;
  }
  ShellAgent::shell_execute(absl::StrFormat("rm -rf %s", path.c_str()));
}

void FSAgent::localfs_mkdir(const string& path) {
  if (path == "") {
    return;
  }

  ShellAgent::shell_execute(absl::StrFormat("mkdir -p %s", path.c_str()));
}

vector<string> FSAgent::localfs_list(const string& path) {
  if (path == "") {
    return {};
  }

  shared_ptr<FILE> pipe;
  pipe = ShellAgent::shell_open_pipe(absl::StrFormat("find %s -type f -maxdepth 1", path.c_str()), "r");
  LineFileReader reader;
  vector<string> list;

  while (reader.getline(&*pipe)) {
    list.push_back(reader.get());
  }

  return list;
}

vector<string> FSAgent::localfs_dirs(const string& path) {
  if (path == "") {
    return {};
  }

  shared_ptr<FILE> pipe;
  pipe = ShellAgent::shell_open_pipe(absl::StrFormat("find %s -type d -maxdepth 1", path.c_str()), "r");
  LineFileReader reader;
  vector<string> list;

  while (reader.getline(&*pipe)) {
    list.push_back(reader.get());
  }

  return list;
}

string FSAgent::localfs_tail(const string& path, const int num) {
  if (path == "") {
    return "";
  }

  return ShellAgent::shell_get_command_output(absl::StrFormat("tail -%d %s ", num, path.c_str()));
}

string FSAgent::localfs_getlines(const string& path, const int start, const int end) {
  if (path == "") {
    return "";
  }

  return ShellAgent::shell_get_command_output(absl::StrFormat("sed -n '%d,%dp' %s ", start, end, path.c_str()));
}

bool FSAgent::localfs_exists(const string& path) {
  string test_f = ShellAgent::shell_get_command_output(absl::StrFormat("[ -f %s ] ; echo $?", path.c_str()));

  absl::StripAsciiWhitespace(&test_f);
  if (test_f == "0") {
    return true;
  }

  string test_d = ShellAgent::shell_get_command_output(absl::StrFormat("[ -d %s ] ; echo $?", path.c_str()));
  absl::StripAsciiWhitespace(&test_d);
  if (test_d == "0") {
    return true;
  }

  return false;
}

bool FSAgent::localfs_touch(const string& path) {
  if(path.size() == 0) {
    return false;
  }

  string test_d = ShellAgent::shell_get_command_output(absl::StrFormat("touch %s ; echo $?", path.c_str()));
  absl::StripAsciiWhitespace(&test_d);
  if (test_d == "0") {
    return true;
  }

  return false;
}

string FSAgent::localfs_md5sum(const string& file) {
  if (file == "") {
    return "";
  }
  string md5_string = ShellAgent::shell_get_command_output(absl::StrFormat("md5sum %s", hdfs_command().c_str(), file.c_str()));
  vector<string> tokens = absl::StrSplit(md5_string, ' ', absl::SkipWhitespace());

  return tokens[0];
}

string FSAgent::localfs_grep(const string& path, const string& segment_str) {
  if (path == "") {
    return "";
  }
  return ShellAgent::shell_get_command_output(absl::StrFormat("cat %s | grep %s ", path.c_str(), segment_str.c_str()));
}

/**************************************  hadoop file system  *************************************/
static string hdfs_command_ = "hadoop fs";
static size_t hdfs_buffer_size_ = 0;

const string& FSAgent::hdfs_command() {
  return hdfs_command_;
}
void FSAgent::hdfs_set_command(const string& x) {
  hdfs_command_ = x;
}

size_t FSAgent::hdfs_buffer_size() {
  return hdfs_buffer_size_;
}
void FSAgent::hdfs_set_buffer_size(size_t x) {
  hdfs_buffer_size_ = x;
}

shared_ptr<FILE> FSAgent::hdfs_open_read(const string& path, const string& converter) {
  string cmd = path;
  if (fs_end_with_internal(cmd, ".gz")) {
    cmd = absl::StrFormat("%s -text \"%s\" | grep -v WARN | grep -v ^$", hdfs_command().c_str(), cmd.c_str());
  } else {
    cmd = absl::StrFormat("%s -cat \"%s\" | grep -v WARN | grep -v ^$", hdfs_command().c_str(), cmd.c_str());
  }

  bool is_pipe = true;
  fs_add_read_converter_internal(&cmd, &is_pipe, converter);
  return fs_open_internal(cmd, is_pipe, "r", hdfs_buffer_size());
}

shared_ptr<FILE> FSAgent::hdfs_open_write(const string& path, const string& converter) {
  string cmd = absl::StrFormat("%s -put - \"%s\"", hdfs_command().c_str(), path.c_str());

  bool is_pipe = true;
  if (fs_end_with_internal(cmd, ".gz\"")) {
    fs_add_write_converter_internal(&cmd, &is_pipe, "gzip");
  }

  fs_add_write_converter_internal(&cmd, &is_pipe, converter);
  return fs_open_internal(cmd, is_pipe, "w", hdfs_buffer_size());
}

int64_t FSAgent::hdfs_file_size(const string& path) {
  string output = ShellAgent::shell_get_command_output(absl::StrFormat("%s -du -s -h %s", hdfs_command().c_str(), path.c_str()));
  vector<string> tokens = absl::StrSplit(output, ' ', absl::SkipWhitespace());

  return (int64_t)(std::stoll(tokens[0]));
}

void FSAgent::hdfs_remove(const string& path) {
  if (path == "") {
    return;
  }

  ShellAgent::shell_execute(absl::StrFormat("%s -rm -r %s; true", hdfs_command().c_str(), path.c_str()));
}

void FSAgent::hdfs_mkdir(const string& path) {
  if (path == "") {
    return;
  }

  ShellAgent::shell_execute(absl::StrFormat("%s -mkdir %s; true", hdfs_command().c_str(), path.c_str()));
}

vector<string> FSAgent::hdfs_list(const string& path) {
  if (path == "") {
    return {};
  }

  shared_ptr<FILE> pipe;
  pipe = ShellAgent::shell_open_pipe(absl::StrFormat("%s -ls %s | ( grep ^[-d] ; [ $? != 2 ] )", hdfs_command().c_str(), path.c_str()), "r");
  LineFileReader reader;
  vector<string> list;

  while (reader.getline(&*pipe)) {
    vector<string> line = absl::StrSplit(reader.get(), ' ', absl::SkipWhitespace());
    CHECK(line.size() == 8);
    list.push_back(line[7]);
  }

  return list;
}

string FSAgent::hdfs_tail(const string& path, const int num) {
  if (path == "") {
    return "";
  }

  return ShellAgent::shell_get_command_output(absl::StrFormat("%s -text %s | tail -%d ", hdfs_command().c_str(), path.c_str(), num));
}

string FSAgent::hdfs_getlines(const string& path, const int start, const int end) {
  if (path == "") {
    return "";
  }

  return ShellAgent::shell_get_command_output(absl::StrFormat("%s -text %s | sed -n '%d,%dp' ", hdfs_command().c_str(), path.c_str(), start, end));
}

bool FSAgent::hdfs_exists(const string& path) {
  string test = ShellAgent::shell_get_command_output(absl::StrFormat("%s -test -e %s ; echo $?", hdfs_command().c_str() , path.c_str()));
  absl::StripAsciiWhitespace(&test);
  if (test == "0") {
    return true;
  }

  return false;
}

bool FSAgent::hdfs_touch(const string& path) {
  string test = ShellAgent::shell_get_command_output(absl::StrFormat("%s -touchz %s ; echo $?", hdfs_command().c_str() , path.c_str()));
  absl::StripAsciiWhitespace(&test);
  if (test == "0") {
    return true;
  }

  return false;
}

vector<string> FSAgent::hdfs_dirs(const string& path) {
  if (path == "") {
    return {};
  }

  shared_ptr<FILE> pipe;
  pipe = ShellAgent::shell_open_pipe(absl::StrFormat("%s -ls %s | ( grep ^d ; [ $? != 2 ] )", hdfs_command().c_str(), path.c_str()), "r");
  LineFileReader reader;
  vector<string> list;

  while (reader.getline(&*pipe)) {
    vector<string> line = absl::StrSplit(reader.get(), ' ', absl::SkipWhitespace());
    CHECK(line.size() == 8);
    list.push_back("hdfs:" + line[7]);
  }

  return list;
}

string FSAgent::hdfs_md5sum(const string& file) {
  if (file == "") {
    return "";
  }
  string md5_string = ShellAgent::shell_get_command_output(absl::StrFormat("%s -cat %s | md5sum", hdfs_command().c_str(), file.c_str()));
  vector<string> tokens = absl::StrSplit(md5_string, ' ', absl::SkipWhitespace());

  return tokens[0];
}

string FSAgent::hdfs_grep(const string& path, const string& segment_str) {
  if (path == "") {
    return "";
  }
  return ShellAgent::shell_get_command_output(absl::StrFormat("%s -cat %s | grep %s", hdfs_command().c_str(), path.c_str(), segment_str.c_str()));
}

/**************************************  common file system **************************************/

shared_ptr<FILE> FSAgent::fs_open(const string& path, const string& mode, const string& converter) {
  if (mode == "r" || mode == "rb") {
    return fs_open_read(path, converter);
  }
  if (mode == "w" || mode == "wb") {
    return fs_open_write(path, converter);
  }
  LOG(FATAL) << "Unknown mode: " << mode;
  return NULL;
}

shared_ptr<FILE> FSAgent::fs_open_read(const string& path, const string& converter) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_open_read(path, converter);
    case 1:
      return hdfs_open_read(path, converter);
    default:
      LOG(FATAL) << "Not supported";
  }
  return NULL;
}

shared_ptr<FILE> FSAgent::fs_open_write(const string& path, const string& converter) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_open_write(path, converter);
    case 1:
      return hdfs_open_write(path, converter);
    default:
      LOG(FATAL) << "Not supported";
  }
  return NULL;
}

int64_t FSAgent::fs_file_size(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_file_size(path);
    case 1:
      return hdfs_file_size(path);
    default:
      LOG(FATAL) << "Not supported";
  }
  return 0;
}

void FSAgent::fs_remove(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_remove(path);
    case 1:
      return hdfs_remove(path);
    default:
      LOG(FATAL) << "Not supported";
  }
  return;
}

void FSAgent::fs_mkdir(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_mkdir(path);
    case 1:
      return hdfs_mkdir(path);
    default:
      LOG(FATAL) << "Not supported";
  }
}

vector<string> FSAgent::fs_list(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_list(path);
    case 1:
      return hdfs_list(path);
    default:
      LOG(FATAL) << "Not supported";
  }
  return {};
}

string FSAgent::fs_tail(const string& path, int num) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_tail(path, num);
    case 1:
      return hdfs_tail(path, num);
    default:
      LOG(FATAL) << "Not supported";
  }
  return "";
}

string FSAgent::fs_getlines(const string& path, int start, int end) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_getlines(path, start, end);
    case 1:
      return hdfs_getlines(path,start, end);
    default:
      LOG(FATAL) << "Not supported";
  }
  return "";
}

bool FSAgent::fs_exists(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_exists(path);
    case 1:
      return hdfs_exists(path);
    default:
      LOG(FATAL) << "Not supported";
  }
  return false;
}

bool FSAgent::fs_touch(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_touch(path);
    case 1:
      return hdfs_touch(path);
    default:
       LOG(FATAL) << "Not supported";
  }
  return false;
}

string FSAgent::fs_md5sum(const string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_md5sum(path);
    case 1:
      return hdfs_md5sum(path);
    default:
       LOG(FATAL) << "Not supported";
  }
  return "";
}

string FSAgent::fs_grep(const string& path, const string& segment_str) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_grep(path, segment_str);
    case 1:
      return hdfs_grep(path, segment_str);
    default:
      LOG(FATAL) << "Not supported";
  }
  return "";
}

/**************************************  private functions  **************************************/
void FSAgent::fs_add_read_converter_internal(string *path, bool *is_pipe, const string& converter) {
  if (converter == "") {
    return;
  }

  if (!(*is_pipe)) {
    (*path) = absl::StrFormat("( %s ) < \"%s\"", converter.c_str(), path->c_str());
    (*is_pipe) = true;
  } else {
    (*path) = absl::StrFormat("%s | %s", path->c_str(), converter.c_str());
  }
}

void FSAgent::fs_add_write_converter_internal(string *path, bool *is_pipe, const string& converter) {
  if (converter == "") {
    return;
  }

  if (!(*is_pipe)) {
    (*path) = absl::StrFormat("( %s ) > \"%s\"", converter.c_str(), path->c_str());
    (*is_pipe) = true;
  } else {
    (*path) = absl::StrFormat("%s | %s", converter.c_str(), path->c_str());
  }
}

shared_ptr<FILE> FSAgent::fs_open_internal(const string& path, const bool is_pipe, const string& mode, const size_t buffer_size) {
  if (!is_pipe) {
    return ShellAgent::shell_open_file(path, mode, buffer_size);
  } else {
    return ShellAgent::shell_open_pipe(path, mode, buffer_size);
  }
}

bool FSAgent::fs_begin_with_internal(const string& path, const string& str) {
  return strncmp(path.c_str(), str.c_str(), str.length()) == 0;
}

bool FSAgent::fs_end_with_internal(const string& path, const string& str) {
  return path.length() >= str.length() && strncmp(&path[path.length() - str.length()], str.c_str(), str.length()) == 0;
}

int FSAgent::fs_select_internal(const string& path) {
  return fs_begin_with_internal(path, "hdfs:") ? 1 : 0;
}

} // namespace model
} // namespace ps

