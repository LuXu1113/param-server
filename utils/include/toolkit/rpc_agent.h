#ifndef UTILS_INCLUDE_TOOLKIT_RPC_AGENT_H_
#define UTILS_INCLUDE_TOOLKIT_RPC_AGENT_H_

#include <vector>
#include <string>
#include <brpc/channel.h>
#include "utils/proto/ps.pb.h"

namespace ps {
namespace toolkit {

struct RPCServerInfo {
  std::string ip_;
  int port_;
};

class RPCAgent {
 public:
  RPCAgent() = delete;

  static int initialize(const std::vector<RPCServerInfo>& rpc_servers);
  static int finalize();
  static int shutdown();

  static int send_to_all(const ParamServerRequest& request, std::vector<ParamServerResponse> *response);
  static int send_to_one(const ParamServerRequest& request, ParamServerResponse *response, size_t server_id);
  static int send_to_one_async(const ParamServerRequest& request, ParamServerResponse *response,
                               size_t server_id, brpc::Controller *cntl, google::protobuf::Closure *done);
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_MPI_AGENT_H_

