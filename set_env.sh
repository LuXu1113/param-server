#!/bin/bash

#-----------------------------   global config   -----------------------------#
source /etc/bashrc
source ~/.bashrc
source ~/.bash_profile

#----------------------------- working directory -----------------------------#
REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)

#-----------------------------  prepare python3  -----------------------------#
cd ${SCRIPT_DIR}/build_tools/python3_setup
mkdir ${SCRIPT_DIR}/build_tools/python3_setup/build
cd ${SCRIPT_DIR}/build_tools/python3_setup/build

../configure --prefix=${SCRIPT_DIR}/build_tools/python3
make
make test
make install

cd ${SCRIPT_DIR}

#-----------------------------   prepare bazel   -----------------------------#
export JAVA_HOME=${SCRIPT_DIR}/build_tools/jdk
export PATH=${SCRIPT_DIR}/build_tools/python3/bin:${PATH}

cd ${SCRIPT_DIR}/build_tools/bazel

env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh

cd ${SCRIPT_DIR}
