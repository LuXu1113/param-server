#ifndef UTILS_INCLUDE_TOOLKIT_SHELL_AGENT_H_
#define UTILS_INCLUDE_TOOLKIT_SHELL_AGENT_H_

#include <memory>
#include <string>

using std::shared_ptr;

namespace ps {
namespace toolkit {

class ShellAgent {
 public:
  ShellAgent() = delete;

  void set_shell_verbose(bool x);

  static shared_ptr<FILE> shell_open_file(const std::string& path, const std::string& mode, const size_t buffer_size = 0);
  static shared_ptr<FILE> shell_open_pipe(const std::string& cmd, const std::string& mode, const size_t buffer_size = 0);
  static void shell_execute(const std::string& cmd);
  static std::string shell_get_command_output(const std::string& cmd);
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_SHELL_AGENT_H_

