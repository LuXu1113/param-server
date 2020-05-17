#!/bin/bash

#-----------------------------   global config   -----------------------------#
source /etc/bashrc
source ~/.bashrc
source ~/.bash_profile

#----------------------------- working directory -----------------------------#
REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)

#-----------------------------  trap debug info ------------------------------#
function debug_info() {
  cmd=`sed -n $2p $1`
  echo "$(date +"%Y/%m/%d %H:%M:%S")][$1:$2][INFO] execute command \"${cmd}\" ..."
}
function error_info() {
  cmd=`sed -n $2p $1`
  echo "$(date +"%Y/%m/%d %H:%M:%S")][$1:$2][ERROR] \"${cmd}\" exit with status $3"
  exit $3
}

# trap 'debug_info ${SCRIPT_NAME} ${LINENO}' DEBUG
trap 'error_info ${SCRIPT_NAME} ${LINENO} $?' ERR

#-----------------------------   prepare bazel   -----------------------------#
export JAVA_HOME=${SCRIPT_DIR}/build_tools/jdk
export PATH=${SCRIPT_DIR}/build_tools/python3/bin:${SCRIPT_DIR}/build_tools/bazel/output:${PATH}

#-----------------------------   make clean   --------------------------------#
# bazel clean --async

#-----------------------------   build examples   ----------------------------#
# bazel build //param_server:param_server -j 12 --compilation_model fastbuild --define with_mesalink=true --define with_glog=true --define with_thrift=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures --sandbox_debug
# bazel build //param_server:param_server -j 12 --compilation_model dbg --define with_mesalink=true --define with_glog=true --define with_thrift=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures --sandbox_debug
# bazel build //param_server:param_server --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures

#---------------------------------    test    --------------------------------#
# bazel test //utils:test_config_manager --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures
# bazel test //utils:test_record --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures
# bazel test //utils:test_sparse_kv_ver1_serialization --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures
# bazel test //utils:test_sparse_embedding_ver1_serialization --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures

#---------------------------------   worker   --------------------------------#
bazel build //param_server:param-server --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures
bazel build //param_server:rtsparse-learner --compilation_mode opt --define with_glog=true --copt -DHAVE_ZLIB=1 --incompatible_disable_deprecated_attr_params=false --verbose_failures

