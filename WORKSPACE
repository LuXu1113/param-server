# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

workspace(name = "ps")

# -----------------------------  libstdc++  ----------------------------#
new_local_repository(
  name = "libstdcxx",
  path = "third_party/libstdc++",
  build_file = "third_party/libstdc++/BUILD",
)

# ------------------------------- tcmalloc -----------------------------#
local_repository(
  name = "com_google_tcmalloc",
  path = "third_party/tcmalloc",
)

# ------------------------------- jemalloc -----------------------------#
new_local_repository(
  name = "jemalloc",
  path = "third_party/jemalloc",
  build_file = "third_party/jemalloc/BUILD",
)

# ------------------------------  gflags  ------------------------------#
local_repository(
  name = "com_github_gflags_gflags",
  path = "third_party/gflags",
)

# ------------------------------   glog   ------------------------------#
local_repository(
  name = "com_github_google_glog",
  path = "third_party/glog",
)

# ------------------------------  abseil  ------------------------------#
local_repository(
  name = "com_google_absl",
  path = "third_party/abseil",
)

# ----------------------------  googletest  ----------------------------#
local_repository(
  name = "com_google_googletest",
  path = "third_party/googletest",
)

# -------------------------------  six  --------------------------------#
new_local_repository(
    name = "six_archive",
    path = "third_party/six",
    build_file = "third_party/six/six.BUILD"
)
bind(
  name = "six",
  actual = "@six_archive//:six",
)

# ----------------------------   protobuf   ----------------------------#
local_repository(
  name = "com_google_protobuf",
  path = "third_party/protobufsource",
)

# -----------------------------  leveldb  ------------------------------#
new_local_repository(
  name = "com_github_google_leveldb",
  path = "third_party/leveldb",
  build_file = "third_party/leveldb/BUILD"
)

# -----------------------------  openssl  ------------------------------#
new_local_repository(
  name = "openssl",
  path = "third_party/openssl",
  build_file = "third_party/openssl/BUILD"
)

# -------------------------------  brpc  -------------------------------#
local_repository(
  name = "com_github_brpc_brpc",
  path = "third_party/brpc",
)

# ------------------------------  openmpi  -----------------------------#
new_local_repository(
  name = "openmpi",
  path = "third_party/openmpi",
  build_file = "third_party/openmpi/BUILD",
)

# ------------------------------  torque  ------------------------------#
new_local_repository(
  name = "torque",
  path = "third_party/torque",
  build_file = "third_party/torque/BUILD",
)

# -----------------------------  libevent  -----------------------------#
new_local_repository(
  name = "libevent",
  path = "third_party/libevent",
  build_file = "third_party/libevent/BUILD",
)

# -----------------------------  yaml-cpp  -----------------------------#
new_local_repository(
  name = "yaml_cpp",
  path = "third_party/yaml-cpp",
  build_file = "third_party/yaml-cpp/BUILD.bazel",
)

# -------------------------------    mkl   -----------------------------#
new_local_repository(
  name = "mkl",
  path = "third_party/mkl",
  build_file = "third_party/mkl/BUILD",
)

# -------------------------------   eigen  -----------------------------#
new_local_repository(
  name = "eigen",
  path = "third_party/eigen",
  build_file = "third_party/eigen/BUILD",
)

