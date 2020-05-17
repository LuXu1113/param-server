#ifndef UTILS_INCLUDE_TOOLKIT_FS_H_
#define UTILS_INCLUDE_TOOLKIT_FS_H_

#include <memory>
#include <vector>
#include <string>

namespace ps {
namespace toolkit {

class FSAgent {
 public:
  FSAgent() = delete;
  FSAgent(const FSAgent&) = delete;
  ~FSAgent() = delete;

  // local file system
  static size_t                   localfs_buffer_size();
  static void                     localfs_set_buffer_size(size_t x);
  static std::shared_ptr<FILE>    localfs_open_read(const std::string& path, const std::string& converter);
  static std::shared_ptr<FILE>    localfs_open_write(const std::string& path, const std::string& converter);
  static int64_t                  localfs_file_size(const std::string& path);
  static void                     localfs_remove(const std::string& path);
  static void                     localfs_mkdir(const std::string& path);
  static std::vector<std::string> localfs_list(const std::string& path);
  static std::vector<std::string> localfs_dirs(const std::string& path);
  static std::string              localfs_tail(const std::string& path, const int num = 1);
  static std::string              localfs_getlines(const std::string& path, const int start, const int end);
  static bool                     localfs_exists(const std::string& path);
  static bool                     localfs_touch(const std::string& path);
  static std::string              localfs_md5sum(const std::string& file);
  static std::string              localfs_grep(const std::string& path, const std::string& segment_str);

  // hadoop file system
  static const std::string&       hdfs_command();
  static void                     hdfs_set_command(const std::string& x);
  static size_t                   hdfs_buffer_size();
  static void                     hdfs_set_buffer_size(size_t x);
  static std::shared_ptr<FILE>    hdfs_open_read(const std::string& path, const std::string& converter);
  static std::shared_ptr<FILE>    hdfs_open_write(const std::string& path, const std::string& converter);
  static int64_t                  hdfs_file_size(const std::string& path);
  static void                     hdfs_remove(const std::string& path);
  static void                     hdfs_mkdir(const std::string& path);
  static std::vector<std::string> hdfs_list(const std::string& path);
  static std::vector<std::string> hdfs_dirs(const std::string& path);
  static std::string              hdfs_tail(const std::string& path, const int num = 1);
  static std::string              hdfs_getlines(const std::string& path, const int start, const int end);
  static bool                     hdfs_exists(const std::string& path);
  static bool                     hdfs_touch(const std::string& path);
  static std::string              hdfs_md5sum(const std::string& file);
  static std::string              hdfs_grep(const std::string& path, const std::string& segment_str);

  // common file system
  static std::shared_ptr<FILE>    fs_open(const std::string& path, const std::string& mode, const std::string& converter = "");
  static std::shared_ptr<FILE>    fs_open_read(const std::string& path, const std::string& converter);
  static std::shared_ptr<FILE>    fs_open_write(const std::string& path, const std::string& converter);
  static int64_t                  fs_file_size(const std::string& path);
  static void                     fs_remove(const std::string& path);
  static void                     fs_mkdir(const std::string& path);
  static std::vector<std::string> fs_list(const std::string& path);
  static std::string              fs_tail(const std::string& path, int num = 1);
  static std::string              fs_getlines(const std::string& path, int start, int end);
  static bool                     fs_exists(const std::string& path);
  static bool                     fs_touch(const std::string& path);
  static std::string              fs_md5sum(const std::string& file);
  static std::string              fs_grep(const std::string& path, const std::string& segment_str);

 private:
  static void fs_add_read_converter_internal(std::string *path, bool *is_pipe, const std::string& converter);
  static void fs_add_write_converter_internal(std::string *path, bool *is_pipe, const std::string& converter);

  static std::shared_ptr<FILE> fs_open_internal(const std::string& path, const bool is_pipe, const std::string& mode, const size_t buffer_size);
  static bool fs_begin_with_internal(const std::string& path, const std::string& str);
  static bool fs_end_with_internal(const std::string& path, const std::string& str);
  static int fs_select_internal(const std::string& path);
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_FS_H_

