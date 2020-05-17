#include "message/types.h"

using std::string;
using std::to_string;

namespace ps {
namespace message {

string errno_to_string(int err_no) {
  string res;

  switch(err_no) {
    case SUCCESS:
      res = "rpc call finished successfully";
      break;
    case UNKNOWN_ERROR:
      res = "rpc call finished with unknown error";
      break;
    case CAN_NOT_ALLOCATE_MEMORY:
      res = "can not allocate memory";
      break;
    case ARRAY_INDEX_OUT_OF_BOUND:
      res = "attempt to access array with an index out of bound";
      break;
    case MESSAGE_TYPE_INVALID:
      res = "invalid RPC request message type";
      break;
    case REGIST_EXISTING_SPARSE_TABLE:
      res = "attempt to register an exist sparse table";
      break;
    case PICK_NONEXISTENT_SPARSE_TABLE:
      res = "attempt to pick a sparse table that does not exist";
      break;
    case REGIST_EXISTING_DENSE_TABLE:
      res = "attempt to register an exist dense table";
      break;
    case PICK_NONEXISTENT_DENSE_TABLE:
      res = "attempt to pick a dense table that does not exist";
      break;
    case REGIST_EXISTING_SUMMARY_TABLE:
      res = "attempt to register an exist summary table";
      break;
    case PICK_NONEXISTENT_SUMMARY_TABLE:
      res = "attempt to pick a summary table that does not exist";
      break;
    case UPDATE_NONEXISTENT_SARSE_FEATURE:
      res = "attempt to update a sparse feature that does not exist";
      break;
    case ASSIGN_NONEXISTENT_SARSE_FEATURE:
      res = "attempt to update a sparse feature that does not exist";
      break;
    case UNKNOWN_OPTIMIZER:
      res = "unknown optimizer";
      break;
    default:
      res = string("err_no: ") + to_string(err_no);
  }

  return res;
}

} // namespace message
} // namespace ps

