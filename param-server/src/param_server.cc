#include <unistd.h>
#include <vector>
#include <string>
#include <gflags/gflags.h>
#include <butil/logging.h>
#include <brpc/server.h>
#include "absl/strings/str_format.h"
#include "utils/proto/ps.pb.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/thread_group.h"
#include "toolkit/data_reader.h"
#include "runtime/config_manager.h"
#include "model/distributed_runner/rtsparse_offline_runner.h"
#include "service_impl.h"

using std::vector;
using std::string;
using ps::toolkit::MPIAgent;
using ps::toolkit::RPCAgent;
using ps::toolkit::RPCServerInfo;
using ps::runtime::ConfigManager;
using ps::model::RTSparseOfflineRunner;

static brpc::Server server;
static ps::param_server::ParamServerServiceImpl ps_service_impl;

int main(int argc, char **argv) {
  /* gflag initialize */
  GFLAGS_NAMESPACE::AllowCommandLineReparsing();
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  /* glog initialize */
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "output";
  FLAGS_max_log_size = 100000;
  FLAGS_logbufsecs = 0;
  FLAGS_minloglevel = google::INFO;
  FLAGS_stderrthreshold = google::ERROR;
  google::InstallFailureSignalHandler();

  /* mkl initialize */
  mkl_set_num_threads(1);

  /* mpi initialize */
  LOG(INFO) << "Step-1: initializing mpi ....";
  MPIAgent::initialize(argc, argv);
  MPIAgent::mpi_barrier_world();
  LOG(INFO) << "Step-1: finished.";

  /* config initialize */
  LOG(INFO) << "Step-2: reading configurations ...";
  bool is_worker = (string("param-server") != google::ProgramInvocationShortName());
  string config_file = absl::StrFormat("conf/%s.yaml", google::ProgramInvocationShortName()).c_str();
  ConfigManager::initialize(config_file, is_worker);

  ps::toolkit::FSAgent::hdfs_set_command(ConfigManager::pick_hdfs_command());
  ps::toolkit::local_thread_group().set_parallel_num(ConfigManager::pick_local_thread_num());
  ps::toolkit::global_write_thread_group().set_parallel_num(ConfigManager::pick_write_thread_num());
  ps::toolkit::DataReader::set_default_capacity(ConfigManager::pick_data_reader_default_capacity());
  ps::toolkit::DataReader::set_default_block_size(ConfigManager::pick_data_reader_default_block_size());
  ps::toolkit::DataReader::set_default_thread_num(ConfigManager::pick_data_reader_default_thread_num());
  MPIAgent::mpi_barrier_world();
  LOG(INFO) << "Step-2: finished.";

  /* start rpc server */
  LOG(INFO) << "Step-3: starting rpc server ...";
  int port = -1;
  if (!is_worker) {
    if (server.AddService(&ps_service_impl, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
      LOG(FATAL) << "Fail to add service.";
    }
    brpc::ServerOptions options;
    options.idle_timeout_sec = -1;

    string local_ip = MPIAgent::mpi_local_ip();
    brpc::PortRange port_range(65000, 75000);
    if (server.Start(local_ip.c_str(), port_range, &options) != 0) {
      LOG(FATAL) << "Fail to start Parameter Server.";
    }
    port = server.listen_address().port;
  }
  MPIAgent::mpi_barrier_world();
  LOG(INFO) << "Step-3: finished.";

  /* broadcast server port */
  LOG(INFO) << "Step-4: broadcasting rpc server ip and port ...";
  vector<int> server_ports;
  MPIAgent::mpi_all_gather(port, server_ports, MPIAgent::mpi_comm_world());
  vector<string> server_ip = MPIAgent::mpi_ip_table_world();
  CHECK(server_ip.size() == server_ports.size());
  vector<RPCServerInfo> rpc_server_info;
  for (int i = 0; i < (int)server_ports.size(); ++i) {
    LOG(INFO) << "  * rank[" << i << "], ip:port = " << server_ip[i] << ":" << server_ports[i];
    if (server_ports[i] != -1) {
      rpc_server_info.push_back({server_ip[i], server_ports[i]});
    }
  }
  ConfigManager::regist_rpc_server_info(rpc_server_info);
  LOG(INFO) << "Step-4: finished.";
  MPIAgent::mpi_barrier_world();

  LOG(INFO) << "Step-5: training ...";
  if (!is_worker) {
    /* serving */
    while (!(ps_service_impl.has_shutdown())) {
      usleep(1000000L);
    }
    server.Stop(50000);
    server.Join();
    LOG(INFO) << "RPC server stopped.";
  } pse {
    /* train */
    RTSparseOfflineRunner runner;
    runner.run();
    LOG(INFO) << "Worker stopped.";
  }
  MPIAgent::mpi_barrier_world();
  LOG(INFO) << "Step-5: finished.";

  LOG(INFO) << "Step-6: exiting ...";
  MPIAgent::finalize();
  LOG(INFO) << "Step-6: finished.";

  LOG(INFO) << "Training fininshed succfully.";

  return 0;
}

