#ifndef UTILS_INCLUDE_MESSAGE_TYPES_H_
#define UTILS_INCLUDE_MESSAGE_TYPES_H_

#include <string>

namespace ps {
namespace message {

// id of RPC request
enum MessageType {
  SPARSE_TABLE_VER1_CREATE = 1,
  SPARSE_TABLE_VER1_SAVE,
  SPARSE_TABLE_VER1_ASSIGN,
  SPARSE_TABLE_VER1_PULL,
  SPARSE_TABLE_VER1_PUSH,
  SPARSE_TABLE_VER1_TIME_DECAY,
  SPARSE_TABLE_VER1_SHRINK,
  SPARSE_TABLE_VER1_FEATURE_NUM,
  EMBEDDING_TABLE_VER1_CREATE,
  EMBEDDING_TABLE_VER1_SAVE,
  EMBEDDING_TABLE_VER1_ASSIGN,
  EMBEDDING_TABLE_VER1_PULL,
  EMBEDDING_TABLE_VER1_PUSH,
  EMBEDDING_TABLE_VER1_TIME_DECAY,
  EMBEDDING_TABLE_VER1_SHRINK,
  EMBEDDING_TABLE_VER1_FEATURE_NUM,
  DENSE_TABLE_VER1_CREATE,
  DENSE_TABLE_VER1_SAVE,
  DENSE_TABLE_VER1_ASSIGN,
  DENSE_TABLE_VER1_PULL,
  DENSE_TABLE_VER1_PUSH,
  DENSE_TABLE_VER1_RESIZE,
  SUMMARY_TABLE_VER1_CREATE,
  SUMMARY_TABLE_VER1_SAVE,
  SUMMARY_TABLE_VER1_ASSIGN,
  SUMMARY_TABLE_VER1_PULL,
  SUMMARY_TABLE_VER1_PUSH,
  SUMMARY_TABLE_VER1_RESIZE,
  SHUTDOWN,
};

// id of RPC return value
enum ErrNo {
  SUCCESS = 0,                           // rpc call finished successfully
  RPC_REMOTE_CALL_FAILED,                // rpc call failed
  CAN_NOT_ALLOCATE_MEMORY,               // can not allocate memory
  ARRAY_INDEX_OUT_OF_BOUND,              // attempt to access array with an index out of bound
  MESSAGE_TYPE_INVALID,                  // invalid RPC request message type
  REGIST_EXISTING_SPARSE_TABLE,          // attempt to register an exist sparse table
  PICK_NONEXISTENT_SPARSE_TABLE,         // attempt to pick a sparse table that does not exist
  REGIST_EXISTING_DENSE_TABLE,           // attempt to register an exist dense table
  PICK_NONEXISTENT_DENSE_TABLE,          // attempt to pick a dense table that does not exist
  REGIST_EXISTING_SUMMARY_TABLE,         // attempt to register an exist dense table
  PICK_NONEXISTENT_SUMMARY_TABLE,        // attempt to pick a dense table that does not exist
  UPDATE_NONEXISTENT_SARSE_FEATURE,      // attempt to update a sparse feature that does not exist
  ASSIGN_NONEXISTENT_SARSE_FEATURE,      // attempt to assign a sparse feature that does not exist
  UNKNOWN_OPTIMIZER,                     // attempt to use unknow optimizer
  UNKNOWN_ERROR,                         // rpc call finished with unknown error
};

std::string errno_to_string(int err_no);

} // namespace message
} // namespace ps

#endif // UTILS_INCLUDE_MESSAGE_TYPES_H_

