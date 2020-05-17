#include "param_table/sparse_embedding_ver1_table.h"

#include <stdio.h>
#include <unistd.h>
#include <atomic>
#include <memory>
#include <algorithm>
#include <butil/logging.h>
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "message/types.h"
#include "toolkit/archive.h"
#include "toolkit/mpi_agent.h"
#include "toolkit/rpc_agent.h"
#include "toolkit/fs_agent.h"
#include "toolkit/thread_group.h"

using std::vector;
using std::string;
using std::atomic;
using std::unique_ptr;
using std::shared_ptr;

using ps::toolkit::BinaryArchive;
using ps::toolkit::MPIAgent;
using ps::toolkit::RPCAgent;
using ps::toolkit::FSAgent;
using ps::runtime::ConfigManager;

namespace ps {
namespace param_table {

static BinaryArchive& operator<<(BinaryArchive& ar, const SparseEmbeddingVer1& val) {
  ar << val.slot_ << val.version_ << val.delta_score_
     << val.silent_days_ << val.count_ << val.ada_d2sum_
     << val.embedding_ << val.ada_g2sum_;
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, SparseEmbeddingVer1& val) {
  ar >> val.slot_ >> val.version_ >> val.delta_score_
     >> val.silent_days_ >> val.count_ >> val.ada_d2sum_
     >> val.embedding_ >> val.ada_g2sum_;
  return ar;
}

static BinaryArchive& operator<<(BinaryArchive& ar, const SparseFeatureVer1& val) {
  ar << val.sign_ << val.slot_;
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, SparseFeatureVer1& val) {
  ar >> val.sign_ >> val.slot_;
  return ar;
}

static BinaryArchive& operator<<(BinaryArchive& ar, const vector<SparseEmbeddingVer1>& p) {
  ar << (size_t)p.size();
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, vector<SparseEmbeddingVer1>& p) {
  p.resize(ar.get<size_t>());
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

static BinaryArchive& operator<<(BinaryArchive& ar, const vector<SparseFeatureVer1>& p) {
  ar << (size_t)p.size();
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}
static BinaryArchive& operator>>(BinaryArchive& ar, vector<SparseFeatureVer1>& p) {
  p.resize(ar.get<size_t>());
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

SparseEmbeddingVer1Shard::SparseEmbeddingVer1Shard() :
  data_(),
  rw_mutex_() {
}

SparseEmbeddingVer1Shard::~SparseEmbeddingVer1Shard() {
}

int SparseEmbeddingVer1Shard::save(const string& file) {
  int ret = ps::message::SUCCESS;

  if (data_.size() > 0) {
    string converter = "";
    shared_ptr<FILE> fd = FSAgent::hdfs_open_write(file, converter);
    for (auto iter : data_) {
      string line;
      rw_mutex_.WriterLock();
      sparse_embedding_ver1_to_string(iter.first, iter.second, &line);
      rw_mutex_.WriterUnlock();

      line = line + string("\n");
      fwrite(line.c_str(), sizeof(char), line.length(), fd.get());
    }
  }

  return ret;
}

int SparseEmbeddingVer1Shard::assign(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(key.size() == value.size());

  rw_mutex_.WriterLock();
  for (size_t i = 0; i < key.size(); ++i) {
    auto iter = data_.find(key[i].sign_);
    if (iter == data_.end()) {
      ret = ps::message::ASSIGN_NONEXISTENT_SARSE_FEATURE;
      break;
    }
  }
  if (ret == ps::message::SUCCESS) {
    for (size_t i = 0; i < key.size(); ++i) {
      auto iter = data_.find(key[i].sign_);
      iter->second = value[i];
    }
  }
  rw_mutex_.WriterUnlock();

  return ret;
}

int SparseEmbeddingVer1Shard::push(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(key.size() == value.size());

  absl::flat_hash_map<SparseKeyVer1, SparseEmbeddingVer1> merge;
  for (size_t i = 0; i < key.size(); ++i) {
    auto iter = merge.find(key[i].sign_);
    if (iter == merge.end()) {
      merge[key[i].sign_] = value[i];
    } else {
      // some kind of feature like "query - title" may make this check fail.
      // CHECK(iter->second.slot_ == value[i].slot_) << "slot-1: " << iter->second.slot_
      //   << ", slots-2: " << value[i].second.slot_;
      ret = sparse_embedding_ver1_merge(&(iter->second), value[i], ConfigManager::pick_training_rule());
    }
  }

  rw_mutex_.WriterLock();
  for (auto i = merge.begin(); i != merge.end(); ++i) {
    auto iter = data_.find(i->first);
    if (iter == data_.end()) {
      ret = ps::message::UPDATE_NONEXISTENT_SARSE_FEATURE;
      break;
    }
  }
  if (ret == ps::message::SUCCESS) {
    for (auto i = merge.begin(); i != merge.end(); ++i) {
      auto iter = data_.find(i->first);
      // some kind of feature like "query - title" may make this check fail.
      // CHECK(key[i].slot_ == iter->second.slot_ && key[i].slot_ == value[i].slot_)
      //   << "sign: " << key[i].sign_ << ", slot-1: " << key[i].slot_
      //   << ", slot-2: " << iter->second.slot_ << ", slot-3: " << value[i].slot_;
      CHECK(iter != data_.end());
      ret = sparse_embedding_ver1_push(&(iter->second), i->second, ConfigManager::pick_training_rule());
    }
  }
  rw_mutex_.WriterUnlock();

  return ret;
}

int SparseEmbeddingVer1Shard::pull(const vector<SparseFeatureVer1>& key, vector<SparseEmbeddingVer1> *value, const bool is_training) {
  int ret = ps::message::SUCCESS;

  value->resize(key.size());
  rw_mutex_.WriterLock();
  for (size_t i = 0; i < key.size(); ++i) {
    auto iter = data_.find(key[i].sign_);
    if (iter == data_.end()) {
      if (is_training) {
        ret = sparse_embedding_ver1_init(&(data_[key[i].sign_]), ConfigManager::pick_training_rule());
        data_[key[i].sign_].slot_ = key[i].slot_;
        (*value)[i] = data_[key[i].sign_];
      } else {
        (*value)[i] = sparse_embedding_ver1_default();
        (*value)[i].slot_ = key[i].slot_;
      }
    } else {
      (*value)[i] = data_[key[i].sign_];
    }
  }
  rw_mutex_.WriterUnlock();

  return ret;
}

int SparseEmbeddingVer1Shard::time_decay() {
  int ret = ps::message::SUCCESS;

  for (auto iter : data_) {
    sparse_embedding_ver1_time_decay(&(iter.second), ConfigManager::pick_training_rule());
  }

  return ret;
}

int SparseEmbeddingVer1Shard::shrink() {
  int ret = ps::message::SUCCESS;

  for (auto iter : data_) {
    if (sparse_embedding_ver1_shrink(iter.second, ConfigManager::pick_training_rule())) {
      data_.erase(iter.first);
    }
  }

  return ret;
}

uint64_t SparseEmbeddingVer1Shard::feature_num() {
  return data_.size();
}

SparseEmbeddingVer1Table::SparseEmbeddingVer1Table() :
  name_(""),
  shard_(31) {
}

SparseEmbeddingVer1Table::SparseEmbeddingVer1Table(const string& name) :
  name_(name),
  shard_(31) {
}

SparseEmbeddingVer1Table::~SparseEmbeddingVer1Table() {
}

const string& SparseEmbeddingVer1Table::name() const {
  return name_;
}

int SparseEmbeddingVer1Table::save(const string& path) {
  int ret = ps::message::SUCCESS;

  size_t mpi_rank = MPIAgent::mpi_rank_group();
  size_t shard_size = shard_.size();

  ps::toolkit::ThreadGroup thread_pool(shard_.size());
  thread_pool.run([this, mpi_rank, shard_size, path](int i) {
      string part_file = path + absl::StrFormat("/part-%05d", mpi_rank * shard_size + i);
      this->shard_[i].save(part_file);
  });
  // for (size_t i = 0; i < shard_.size(); ++i) {
  //   string part_file = path + absl::StrFormat("/part-%05d", MPIAgent::mpi_rank_group() * shard_.size() + i);
  //   shard_[i].save(part_file);
  // }

  return ret;
}

int SparseEmbeddingVer1Table::assign(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(key.size() == value.size());
  size_t bin_num = shard_.size();
  vector<vector<SparseFeatureVer1> > tmp_key(bin_num);
  vector<vector<SparseEmbeddingVer1> >   tmp_value(bin_num);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t bin = absl::Hash<SparseKeyVer1>()(key[i].sign_) % bin_num;
    size_t bin = key[i].sign_ % bin_num;
    tmp_key[bin].push_back(key[i]);
    tmp_value[bin].push_back(value[i]);
  }

  for (size_t i = 0; i < bin_num; ++i) {
    ret = shard_[i].assign(tmp_key[i], tmp_value[i]);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }

  return ret;
}

int SparseEmbeddingVer1Table::push(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) {
  int ret = ps::message::SUCCESS;

  CHECK(key.size() == value.size());
  size_t bin_num = shard_.size();
  vector<vector<SparseFeatureVer1> > tmp_key(bin_num);
  vector<vector<SparseEmbeddingVer1> > tmp_value(bin_num);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t bin = absl::Hash<SparseKeyVer1>()(key[i].sign_) % bin_num;
    size_t bin = key[i].sign_ % bin_num;
    tmp_key[bin].push_back(key[i]);
    tmp_value[bin].push_back(value[i]);
  }

  for (size_t i = 0; i < bin_num; ++i) {
    ret = shard_[i].push(tmp_key[i], tmp_value[i]);
    if (ps::message::SUCCESS != ret) {
      break;
    }
  }

  return ret;
}

int SparseEmbeddingVer1Table::pull(const vector<SparseFeatureVer1>& key, vector<SparseEmbeddingVer1> *value, const bool is_training) {
  int ret = ps::message::SUCCESS;

  value->resize(key.size());
  size_t bin_num = shard_.size();
  vector<vector<SparseFeatureVer1> > tmp_key(bin_num);
  vector<vector<size_t> > tmp_index(bin_num);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t bin = absl::Hash<SparseKeyVer1>()(key[i].sign_) % bin_num;
    size_t bin = key[i].sign_ % bin_num;
    tmp_key[bin].push_back(key[i]);
    tmp_index[bin].push_back(i);
  }

  for (size_t i = 0; i < bin_num; ++i) {
    vector<SparseEmbeddingVer1> tmp_value;
    ret = shard_[i].pull(tmp_key[i], &(tmp_value), is_training);
    if (ps::message::SUCCESS != ret) {
      break;
    }

    for (size_t j = 0; j < tmp_value.size(); ++j) {
      (*value)[tmp_index[i][j]] = tmp_value[j];
    }
  }

  return ret;
}

int SparseEmbeddingVer1Table::time_decay() {
  int ret = ps::message::SUCCESS;

  ps::toolkit::ThreadGroup thread_pool(shard_.size());
  thread_pool.run([this](int i) {
    this->shard_[i].time_decay();
  });

  return ret;
}

int SparseEmbeddingVer1Table::shrink() {
  int ret = ps::message::SUCCESS;

  ps::toolkit::ThreadGroup thread_pool(shard_.size());
  thread_pool.run([this](int i) {
    this->shard_[i].shrink();
  });

  return ret;
}

uint64_t SparseEmbeddingVer1Table::feature_num() {
  uint64_t feature_num = 0;
  for (auto iter = shard_.begin(); iter != shard_.end(); ++iter) {
    feature_num += iter->feature_num();
  }
  return feature_num;
}

SparseEmbeddingVer1TableServer::SparseEmbeddingVer1TableServer() :
  tables_() {
}

SparseEmbeddingVer1TableServer::~SparseEmbeddingVer1TableServer() {
  for (auto iter = tables_.begin(); iter != tables_.end(); ++iter) {
    delete (iter->second);
  }
}

int SparseEmbeddingVer1TableServer::create(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;

  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = ps::message::REGIST_EXISTING_SPARSE_TABLE;
  } else {
    tables_[table_name] = new SparseEmbeddingVer1Table(table_name);
    if (NULL == tables_[table_name]) {
      ret = ps::message::CAN_NOT_ALLOCATE_MEMORY;
    }
  }
  response->set_return_value(ret);
  LOG(INFO) << "create embedding table: " << table_name << ", ret = " << ret;

  return ret;
}

int SparseEmbeddingVer1TableServer::save(const ParamServerRequest& request, ParamServerResponse *response) const {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = iter->second->save(request.message());
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }
  LOG(INFO) << "save embedding table: " << table_name << ", path = " << request.message() << ", ret = " << ret;

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::assign(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    vector<SparseFeatureVer1> new_key;
    ar >> new_key;
    vector<SparseEmbeddingVer1> new_value;
    ar >> new_value;

    ret = iter->second->assign(new_key, new_value);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::push(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    vector<SparseFeatureVer1> push_key;
    ar >> push_key;
    vector<SparseEmbeddingVer1> push_value;
    ar >> push_value;

    ret = iter->second->push(push_key, push_value);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::pull(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    string message = request.message();
    BinaryArchive ar;
    ar.set_read_buffer(message);

    vector<SparseFeatureVer1> pull_key;
    ar >> pull_key;

    bool is_training = request.is_training();

    vector<SparseEmbeddingVer1> pull_value;
    ret = iter->second->pull(pull_key, &pull_value, is_training);
    if (ret == ps::message::SUCCESS) {
      CHECK(pull_key.size() == pull_value.size());
      BinaryArchive oar;
      oar << pull_value;

      string message;
      oar.release(&message);

      response->set_message(message);
    }
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::time_decay(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = iter->second->time_decay();
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }
  LOG(INFO) << "embedding table time decay: " << table_name << ", ret = " << ret;

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::shrink(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    ret = iter->second->shrink();
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }
  LOG(INFO) << "shrink embedding table: " << table_name << ", ret = " << ret;

  response->set_return_value(ret);
  return ret;
}

int SparseEmbeddingVer1TableServer::feature_num(const ParamServerRequest& request, ParamServerResponse *response) {
  int ret = ps::message::SUCCESS;
  const string& table_name = request.table_name();
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    uint64_t feature_num = iter->second->feature_num();
    BinaryArchive oar;
    oar << feature_num;

    string message;
    oar.release(&message);

    response->set_message(message);
  } else {
    ret = ps::message::PICK_NONEXISTENT_SPARSE_TABLE;
  }

  response->set_return_value(ret);
  return ret;
}

SparseEmbeddingVer1TableClient::SparseEmbeddingVer1TableClient() :
  name_("") {
}

const string& SparseEmbeddingVer1TableClient::name() const {
  return name_;
}

int SparseEmbeddingVer1TableClient::create(const string& name) {
  int ret = 0;
  name_ = name;

  DLOG(INFO) << "create embedding table: " << name_;
  if (MPIAgent::mpi_rank_group() == 0) {
    ParamServerRequest request;
    vector<ParamServerResponse> response;
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_CREATE);
    request.set_table_name(name_);

    ret = RPCAgent::send_to_all(request, &response);
    if (0 != ret) {
      LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_CREATE, ret = " << ret;
    } else {
      for (size_t i = 0; i < response.size(); ++i) {
        ret = response[i].return_value();
        if (ps::message::SUCCESS != ret) {
          LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret)
                     << ", message_type = " << request.message_type()
                     << ", table_name = " << request.table_name();
          continue;
        }
      }
    }
  }

  return ret;
}

static void handle_async_save_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(FATAL) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";
    }
  }
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SparseEmbeddingVer1TableClient::save(const string& path) const {
  int ret = 0;

  LOG(INFO) << "save embedding table: " << name_ << ", path = " << path;
  if (MPIAgent::mpi_rank_group() == 0) {
    size_t mpi_size = MPIAgent::mpi_size_group();
    atomic<int> count(mpi_size);

    ParamServerRequest request;
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_SAVE);
    request.set_table_name(name_);
    request.set_message(path);

    for (size_t i = 0; i < mpi_size; ++i) {
      ParamServerResponse *response = new ParamServerResponse();
      brpc::Controller *cntl = new brpc::Controller();
      google::protobuf::Closure *done = brpc::NewCallback(&handle_async_save_response, cntl, response, i, &count);

      ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
      if (0 != ret) {
        LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_SAVE, ret = " << ret;
        continue;
      }
    }

    while (count > 0) {
      usleep(5000);
    }
  }
  LOG(INFO) << "finish save embedding table: " << name_ << ", path = " << path;

  return ret;
}
static void handle_async_assign_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(ERROR) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";
    }
  }

  return;
}

int SparseEmbeddingVer1TableClient::assign(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) const {
  int ret = 0;
  CHECK(key.size() == value.size());

  size_t mpi_size = MPIAgent::mpi_size_group();

  DLOG(INFO) << "assign embedding table: " << name_;
  vector<vector<SparseFeatureVer1> > tmp_key;
  vector<vector<SparseEmbeddingVer1> > tmp_value;
  tmp_key.resize(mpi_size);
  tmp_value.resize(mpi_size);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t partition_id = absl::Hash<SparseKeyVer1>()(key[i].sign_) % mpi_size;
    size_t partition_id = key[i].sign_ % mpi_size;
    tmp_key[partition_id].push_back(key[i]);
    tmp_value[partition_id].push_back(value[i]);
  }

  for (size_t i = 0; i < mpi_size; ++i) {
    ParamServerRequest request;
    ParamServerResponse *response = new ParamServerResponse();
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_ASSIGN);
    request.set_table_name(name_);

    BinaryArchive ar;
    ar << tmp_key[i] << tmp_value[i];

    string message;
    ar.release(&message);

    request.set_message(message);

    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_assign_response, cntl, response, i);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
    if (0 != ret) {
      LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_ASSIGN, ret = " << ret;
      continue;
    }
  }

  return ret;
}

static void handle_async_push_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id) {
  // std::unique_ptr makes sure cntl/response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(ERROR) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";
    }
  }

  return;
}

int SparseEmbeddingVer1TableClient::push(const vector<SparseFeatureVer1>& key, const vector<SparseEmbeddingVer1>& value) const {
  int ret = 0;

  CHECK(key.size() == value.size());
  size_t mpi_size = MPIAgent::mpi_size_group();

  DLOG(INFO) << "push embedding table: " << name_;
  vector<vector<SparseFeatureVer1> > tmp_key;
  vector<vector<SparseEmbeddingVer1> > tmp_value;
  tmp_key.resize(mpi_size);
  tmp_value.resize(mpi_size);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t partition_id = absl::Hash<SparseKeyVer1>()(key[i].sign_) % mpi_size;
    size_t partition_id = key[i].sign_ % mpi_size;
    tmp_key[partition_id].push_back(key[i]);
    tmp_value[partition_id].push_back(value[i]);
  }

  for (size_t i = 0; i < mpi_size; ++i) {
    ParamServerRequest request;
    ParamServerResponse *response = new ParamServerResponse();
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_PUSH);
    request.set_table_name(name_);

    BinaryArchive ar;
    ar << tmp_key[i] << tmp_value[i];

    string message;
    ar.release(&message);
    request.set_message(message);

    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_push_response, cntl, response, i);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
    if (0 != ret) {
      LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_PUSH, ret = " << ret;
      continue;
    }
  }

  return ret;
}

void handle_async_pull_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id,
    vector<vector<uint32_t> > *tmp_mapping, vector<SparseEmbeddingVer1> *value, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(ERROR) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";

      vector<SparseEmbeddingVer1> tmp_value;
      BinaryArchive oar;
      oar.set_read_buffer(response->message());
      oar >> tmp_value;

      CHECK(tmp_value.size() == (*tmp_mapping)[server_id].size());
      for (size_t i = 0; i < tmp_value.size(); ++i) {
        (*value)[(*tmp_mapping)[server_id][i]] = tmp_value[i];
      }
    }
  }
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SparseEmbeddingVer1TableClient::pull(const vector<SparseFeatureVer1>&key, vector<SparseEmbeddingVer1> *value, const bool is_training) const {
  int ret = 0;

  value->resize(key.size());
  size_t mpi_size = MPIAgent::mpi_size_group();
  atomic<int> count(mpi_size);

  DLOG(INFO) << "pull embedding table: " << name_;
  vector<vector<SparseFeatureVer1> > tmp_key;
  vector<vector<uint32_t> > tmp_mapping;
  tmp_key.resize(mpi_size);
  tmp_mapping.resize(mpi_size);

  for (size_t i = 0; i < key.size(); ++i) {
    // size_t partition_id = absl::Hash<SparseKeyVer1>()(key[i].sign_) % mpi_size;
    size_t partition_id = key[i].sign_ % mpi_size;
    tmp_key[partition_id].push_back(key[i]);
    tmp_mapping[partition_id].push_back(i);
  }

  for (size_t i = 0; i < mpi_size; ++i) {
    BinaryArchive ar;
    ar << tmp_key[i];

    string message;
    ar.release(&message);

    ParamServerRequest request;
    ParamServerResponse *response = new ParamServerResponse();
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_PULL);
    request.set_table_name(name_);
    request.set_message(message);
    request.set_is_training(is_training);

    brpc::Controller *cntl = new brpc::Controller();
    google::protobuf::Closure *done = brpc::NewCallback(&handle_async_pull_response, cntl, response, i,
      &tmp_mapping, value, &count);

    ret = RPCAgent::send_to_one_async(request, response, i, cntl,  done);
    if (0 != ret) {
      LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_PULL, ret = " << ret;
      continue;
    }
  }

  while (count > 0) {
    usleep(5000);
  }

  return ret;
}

static void handle_async_time_decay_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(FATAL) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";
    }
  }
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SparseEmbeddingVer1TableClient::time_decay() const {
  int ret = 0;

  LOG(INFO) << "embedding table time decay: " << name_;
  if (MPIAgent::mpi_rank_group() == 0) {
    size_t mpi_size = MPIAgent::mpi_size_group();
    atomic<int> count(mpi_size);

    ParamServerRequest request;
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_TIME_DECAY);
    request.set_table_name(name_);

    for (size_t i = 0; i < mpi_size; ++i) {
      ParamServerResponse *response = new ParamServerResponse();
      brpc::Controller *cntl = new brpc::Controller();
      google::protobuf::Closure *done = brpc::NewCallback(&handle_async_time_decay_response, cntl, response, i, &count);

      ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
      if (0 != ret) {
        LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_TIME_DECAY, ret = " << ret;
        continue;
      }
    }

    while (count > 0) {
      usleep(5000);
    }
  }
  LOG(INFO) << "finish embedding table time decay: " << name_;

  return ret;
}

static void handle_async_shrink_response(brpc::Controller *cntl, ParamServerResponse *response, size_t server_id, atomic<int> *count) {
  // std::unique_ptr makes sure response will be deleted before returning.
  unique_ptr<brpc::Controller> cntl_guard(cntl);
  unique_ptr<ParamServerResponse> response_guard(response);

  if (cntl->Failed()) {
    LOG(FATAL) << "remote_call to " << cntl->remote_side() << " fail, error text is:" << cntl->ErrorText();
  } else {
    int ret = response->return_value();
    if (ps::message::SUCCESS != ret) {
      LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret);
    } else {
      DLOG(INFO) << "Received response from " << cntl->remote_side()
                 << ": " << response->message() << " (attached = " << cntl->response_attachment() << ")"
                 << ", latency = " << cntl->latency_us() << "us";
    }
  }
  if (NULL != count) {
    --(*count);
  }

  return;
}

int SparseEmbeddingVer1TableClient::shrink() const {
  int ret = 0;

  LOG(INFO) << "shrink embedding table: " << name_;
  if (MPIAgent::mpi_rank_group() == 0) {
    size_t mpi_size = MPIAgent::mpi_size_group();
    atomic<int> count(mpi_size);

    ParamServerRequest request;
    request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_SHRINK);
    request.set_table_name(name_);

    for (size_t i = 0; i < mpi_size; ++i) {
      ParamServerResponse *response = new ParamServerResponse();
      brpc::Controller *cntl = new brpc::Controller();
      google::protobuf::Closure *done = brpc::NewCallback(&handle_async_shrink_response, cntl, response, i, &count);

      ret = RPCAgent::send_to_one_async(request, response, i, cntl, done);
      if (0 != ret) {
        LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_SHRINK, ret = " << ret;
        continue;
      }
    }

    while (count > 0) {
      usleep(5000);
    }
  }
  LOG(INFO) << "finish shrink embedding table: " << name_;

  return ret;
}

uint64_t SparseEmbeddingVer1TableClient::feature_num() const {
  uint64_t feature_num = 0;
  int ret = 0;

  DLOG(INFO) << "embedding table feature num: " << name_;
  ParamServerRequest request;
  vector<ParamServerResponse> response;
  request.set_message_type(ps::message::EMBEDDING_TABLE_VER1_FEATURE_NUM);
  request.set_table_name(name_);

  ret = RPCAgent::send_to_all(request, &response);
  if (0 != ret) {
    LOG(FATAL) << "rpc call EMBEDDING_TABLE_VER1_CREATE, ret = " << ret;
  } else {
    for (size_t i = 0; i < response.size(); ++i) {
      ret = response[i].return_value();
      if (ps::message::SUCCESS != ret) {
        LOG(FATAL) << "ErrNo = " << ps::message::errno_to_string(ret)
                   << ", message_type = " << request.message_type()
                   << ", table_name = " << request.table_name();
        continue;
      }

      uint64_t tmp_value;
      BinaryArchive oar;
      oar.set_read_buffer(response[i].message());
      oar >> tmp_value;
      feature_num += tmp_value;
    }
  }

  return feature_num;
}

} // namespace param_table
} // namespace ps

