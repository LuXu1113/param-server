#include "toolkit/mpi_agent.h"

#include <stdlib.h>
#include <memory.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <mpi.h>

#include <vector>
#include <string>

using std::vector;
using std::string;
using std::map;

namespace ps {
namespace toolkit {

static string local_ip_;

static int mpi_rank_world_ = 0;
static int mpi_size_world_ = 1;
static vector<string> mpi_ip_table_world_;
static MPI_Comm mpi_comm_world_;

static MPI_Group mpi_group_;
static int mpi_rank_group_ = 0;
static int mpi_size_group_ = 1;
static vector<string> mpi_ip_table_group_;
static MPI_Comm mpi_comm_group_;

const string& MPIAgent::mpi_local_ip() {
  return local_ip_;
}

int MPIAgent::mpi_rank_world() {
  return mpi_rank_world_;
}
int MPIAgent::mpi_size_world() {
  return mpi_size_world_;
}
const vector<string>& MPIAgent::mpi_ip_table_world() {
  return mpi_ip_table_world_;
}
MPI_Comm MPIAgent::mpi_comm_world() {
  return mpi_comm_world_;
}
void MPIAgent::mpi_barrier_world() {
  MPI_Barrier(mpi_comm_world_);
}

int MPIAgent::mpi_size_group() {
  return mpi_size_group_;
}
int MPIAgent::mpi_rank_group() {
  return mpi_rank_group_;
}
const vector<string>& MPIAgent::mpi_ip_table_group() {
  return mpi_ip_table_group_;
}
MPI_Comm MPIAgent::mpi_comm_group() {
  return mpi_comm_group_;
}
void MPIAgent::mpi_barrier_group() {
  MPI_Barrier(mpi_comm_group_);
}

string MPIAgent::mpi_get_local_ip_internal() {
  string local_ip = "";
  int ret = 0;
  int socket_fd = -1;
  char *buffer = NULL;

  buffer = (char *) malloc(512);
  CHECK(NULL != buffer) << "malloc() fail, buffer = " << buffer;

  socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  CHECK(socket_fd >= 0) << "socket() fail, socket_fd = " << socket_fd;

  struct ifconf if_conf;
  if_conf.ifc_len = 512;
  if_conf.ifc_buf = buffer;
  ret = ioctl(socket_fd, SIOCGIFCONF, &if_conf);
  CHECK (ret >= 0) << "ioctl() fail, ret = " << ret;

  struct ifreq *if_req = (struct ifreq *)buffer;
  for (int i = 0; i < int(if_conf.ifc_len / sizeof(struct ifreq)); ++i) {
    local_ip = inet_ntoa(((struct sockaddr_in*)&if_req->ifr_addr)->sin_addr);

    if ("127.0.0.1" != local_ip) {
      break;
    } else {
      local_ip = "";
    }

    ++if_req;
  }

  CHECK(local_ip != "") << "mpi_get_local_ip() fail.";
  return local_ip;
}

int MPIAgent::initialize(int argc, char **argv) {
  int ret = 0;

  ret = MPI_Init(&argc, &argv);
  CHECK(0 == ret) << "MPI_Init() fail, ret = " << ret;

  // old versions of openmpi changes SIGCHLD handler in MPI_Init, fix it here.
  struct sigaction sigaction_new;
  memset(&sigaction_new, 0, sizeof(sigaction_new));
  sigaction_new.sa_handler = SIG_DFL;
  ret = sigaction(SIGCHLD, &sigaction_new, NULL);
  CHECK(0 == ret) << "sigaction() fail, ret = " << ret;

  // split nodes to 2 parts: servers and workers
  mpi_comm_world_ = MPI_COMM_WORLD;
  ret = MPI_Comm_rank(mpi_comm_world_, &mpi_rank_world_);
  CHECK(0 == ret) << "MPI_Comm_rank() fail, ret = " << ret;
  ret = MPI_Comm_size(mpi_comm_world_, &mpi_size_world_);
  CHECK(0 == ret) << "MPI_Comm_size() fail, ret = " << ret;

  string argv0 = argv[0];
  std::vector<std::string> bin_vec;
  mpi_all_gather(argv0, bin_vec, mpi_comm_world_);

  std::vector<int> rank_arr;
  for (int i = 0; i < mpi_size_world_; ++i) {
    if (bin_vec[mpi_rank_world_] == bin_vec[i]) {
      rank_arr.push_back(i);
    }
  }

  MPI_Group orig_group;
  ret = MPI_Barrier(mpi_comm_world_);
  CHECK(0 == ret) << "MPI_Barrier() fail, ret = " << ret;
  ret = MPI_Comm_group(mpi_comm_world_, &orig_group);
  CHECK(0 == ret) << "MPI_Comm_group() fail, ret = " << ret;
  ret = MPI_Group_incl(orig_group, rank_arr.size(), &rank_arr[0], &mpi_group_);
  CHECK(0 == ret) << "MPI_Group_incl() fail, ret = " << ret;
  ret = MPI_Comm_create(mpi_comm_world_, mpi_group_, &mpi_comm_group_);
  CHECK(0 == ret) << "MPI_Comm_create() fail, ret = " << ret;
  ret = MPI_Comm_rank(mpi_comm_group_, &mpi_rank_group_);
  CHECK(0 == ret) << "MPI_Comm_rank() fail, ret = " << ret;
  ret = MPI_Comm_size(mpi_comm_group_, &mpi_size_group_);
  CHECK(0 == ret) << "MPI_Comm_size() fail, ret = " << ret;

  local_ip_ = mpi_get_local_ip_internal();
  mpi_all_gather(local_ip_, mpi_ip_table_world_, mpi_comm_world_);
  mpi_all_gather(local_ip_, mpi_ip_table_group_, mpi_comm_group_);

  for (int i = 0; i < mpi_size_world_; ++i) {
    LOG(INFO) << "executable file of rank [" << i << "] = " << bin_vec[i] << ", ip = " << mpi_ip_table_world_[i];
  }

  LOG(INFO) << "this process belongs to mpi group: {";
  for (int i = 0; i < mpi_size_group_; ++i) {
    LOG(INFO) << "    world rank = " << rank_arr[i]
              << "    executable file = " << bin_vec[rank_arr[i]]
              << "    ip = " << mpi_ip_table_group_[i];
  }
  LOG(INFO) << "}, group rank = " << mpi_rank_group_ << ", world rank = " << mpi_rank_world_;

  return ret;
}

int MPIAgent::finalize() {
  int ret = MPI_Finalize();
  CHECK(0 == ret) << "MPI_Finalize fail, ret = " << ret;

  return ret;
}

} // namespace toolkit
} // namespace ps

