#ifndef UTILS_INCLUDE_TOOLKIT_MPI_AGENT_H_
#define UTILS_INCLUDE_TOOLKIT_MPI_AGENT_H_

#include <mpi.h>
#include <vector>
#include <string.h>
#include <butil/logging.h>
#include "absl/hash/hash.h"
#include "toolkit/archive.h"

namespace ps {
namespace toolkit {

template<class T>
struct mpi_type_trait {
  static MPI_Datatype type() {
    return (MPI_Datatype) - 1;
  }
};

template<>
struct mpi_type_trait<double> {
  static MPI_Datatype type() {
    return MPI_DOUBLE;
  }
};

template<>
struct mpi_type_trait<float> {
  static MPI_Datatype type() {
    return MPI_FLOAT;
  }
};

template<>
struct mpi_type_trait<int32_t> {
  static MPI_Datatype type() {
    return MPI_INT;
  }
};

template<>
struct mpi_type_trait<uint32_t> {
  static MPI_Datatype type() {
    return MPI_UNSIGNED;
  }
};

template<>
struct mpi_type_trait<int64_t> {
  static MPI_Datatype type() {
    return MPI_LONG_LONG;
  }
};

template<>
struct mpi_type_trait<uint64_t> {
  static MPI_Datatype type() {
    return MPI_UNSIGNED_LONG_LONG;
  }
};

template<>
struct mpi_type_trait<long long> {
  static MPI_Datatype type() {
    return MPI_LONG_LONG;
  }
};

template<>
struct mpi_type_trait<unsigned long long> {
  static MPI_Datatype type() {
    return MPI_UNSIGNED_LONG_LONG;
  }
};

class MPIAgent {
 public:
  MPIAgent() = delete;

  static int initialize(int argc, char **argv);
  static int finalize();
  static const std::string &mpi_local_ip();

  static int mpi_size_world();
  static int mpi_rank_world();
  static const std::vector<std::string>& mpi_ip_table_world();
  static void mpi_barrier_world();
  static MPI_Comm mpi_comm_world();

  static int mpi_size_group();
  static int mpi_rank_group();
  static const std::vector<std::string>& mpi_ip_table_group();
  static void mpi_barrier_group();
  static MPI_Comm mpi_comm_group();

  template<class T>
  static void mpi_bcast_group(T* p, int count, int root) {
    ps::toolkit::BinaryArchive ar;
    int len = 0;

    if (mpi_rank_group() == root) {
      for (int i = 0; i < count; ++i) {
        ar << p[i];
      }
      len = ar.length();
    }

    CHECK(0 == MPI_Bcast(&len, 1, MPI_INT, root, mpi_comm_group()));
    ar.resize(len);
    ar.set_cursor(ar.buffer());
    CHECK(0 == MPI_Bcast(ar.buffer(), len, MPI_BYTE, root, mpi_comm_group()));

    for (int i = 0; i < count; ++i) {
      ar >> p[i];
    }
  }

  template<class T>
  static void mpi_bcast_world(T* p, int count, int root) {
    ps::toolkit::BinaryArchive ar;
    int len = 0;

    if (mpi_rank_world() == root) {
      for (int i = 0; i < count; ++i) {
        ar << p[i];
      }
      len = ar.length();
    }

    CHECK(0 == MPI_Bcast(&len, 1, MPI_INT, root, mpi_comm_world()));
    ar.resize(len);
    ar.set_cursor(ar.buffer());
    CHECK(0 == MPI_Bcast(ar.buffer(), len, MPI_BYTE, root, mpi_comm_world()));

    for (int i = 0; i < count; ++i) {
      ar >> p[i];
    }
  }

  template<class T>
  static void mpi_check_consistency_group(const T* p, int count) {
    ps::toolkit::BinaryArchive ar;

    for (int i = 0; i < count; i++) {
      ar << p[i];
    }

    std::string str;
    ar.release(&str);
    uint64_t hash_code = absl::Hash<std::string>()(str);
    uint64_t root_hash_code = hash_code;
    CHECK(0 == MPI_Bcast(&root_hash_code, 1, mpi_type_trait<uint64_t>::type(), 0, mpi_comm_group()));
    CHECK(root_hash_code == hash_code);
  }

  template<class T>
  static void mpi_check_consistency_world(const T* p, int count) {
    ps::toolkit::BinaryArchive ar;

    for (int i = 0; i < count; i++) {
      ar << p[i];
    }

    std::string str;
    ar.release(&str);

    uint64_t hash_code = absl::Hash<std::string>()(str);
    uint64_t root_hash_code = hash_code;
    CHECK(0 == MPI_Bcast(&root_hash_code, 1, mpi_type_trait<uint64_t>::type(), 0, mpi_comm_world()));
    CHECK(root_hash_code == hash_code);
  }

  template<class T>
  static T mpi_allreduce_group(T x, MPI_Op op) {
    T tot;
    CHECK(0 == MPI_Allreduce(&x, &tot, 1, mpi_type_trait<T>::type(), op, mpi_comm_group()));
    return tot;
  }

  template<class T>
  static T mpi_allreduce_world(T x, MPI_Op op) {
    T tot;
    CHECK(0 == MPI_Allreduce(&x, &tot, 1, mpi_type_trait<T>::type(), op, mpi_comm_world()));
    return tot;
  }

  template<class T>
  static void mpi_all_gather(T p, std::vector<T>& v, MPI_Comm comm = MPI_COMM_WORLD) {
    ps::toolkit::BinaryArchive ar;
    int mr = 0;
    int ms = 0;
    PCHECK(0 == MPI_Comm_rank(comm, &mr));
    PCHECK(0 == MPI_Comm_size(comm, &ms));
    std::vector<int> len_list(ms);
    std::vector<int> dis_list(ms);
    v.resize(ms);
    if (mpi_type_trait<T>::type() != (MPI_Datatype) - 1) {
      v[mr] = p;
      MPI_Datatype tp = mpi_type_trait<T>::type();
      PCHECK(0 == MPI_Allgather(MPI_IN_PLACE, 0, tp, &v[0], 1, tp, comm));
      return;
    }

    ar << p;
    len_list[mr] = ar.length();
    PCHECK(0 == MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, &len_list[0], 1, MPI_INT, comm));

    int sum = 0;

    for (int i = 0; i < ms; ++i) {
      dis_list[i] = sum;
      sum += len_list[i];
    }

    ps::toolkit::BinaryArchive oar;
    oar.resize(sum);
    PCHECK(0 == MPI_Allgatherv((void*)ar.buffer(), ar.length(), MPI_BYTE,
          oar.buffer(), &len_list[0], &dis_list[0], MPI_BYTE, comm));

    ps::toolkit::BinaryArchive ar_reader;
    for (int i = 0; i < ms; ++i) {
      ar_reader.set_read_buffer(oar.buffer() + dis_list[i], len_list[i]);
      ar_reader >> v[i];
    }
  }

 private:
  static std::string mpi_get_local_ip_internal();
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_MPI_AGENT_H_

