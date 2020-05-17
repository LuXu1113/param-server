# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
  name = "ps_cc_proto",
  srcs = [
    "proto/ps.proto",
  ],
  cc_libs=["@com_google_protobuf//:protobuf"],
  protoc="@com_google_protobuf//:protoc",
  default_runtime="@com_google_protobuf//:protobuf",
  visibility=["//visibility:public"],
)

COPTS = [
  "--std=c++11",
  "-Wno-unused-parameter",
  "-fno-omit-frame-pointer",
  "-fPIC",
];

cc_library(
  name = "message",
  srcs = [
    "include/message/types.h",
    "src/message/types.cc",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  visibility = ["//visibility:public"],
)

cc_library(
  name = "toolkit",
  srcs = [
    "include/toolkit/archive.h",
    "include/toolkit/config.h",
    "include/toolkit/channel.h",
    "include/toolkit/factory.h",
    "include/toolkit/fs_agent.h",
    "include/toolkit/mpi_agent.h",
    "include/toolkit/rpc_agent.h",
    "include/toolkit/operating_log.h",
    "include/toolkit/shell_agent.h",
    "include/toolkit/string_agent.h",
    "include/toolkit/semaphore.h",
    "include/toolkit/managed_thread.h",
    "include/toolkit/thread_group.h",
    "include/toolkit/data_reader.h",
    "include/toolkit/parallel_data_processor.h",
    "src/toolkit/archive.cc",
    "src/toolkit/config.cc",
    "src/toolkit/fs_agent.cc",
    "src/toolkit/mpi_agent.cc",
    "src/toolkit/rpc_agent.cc",
    "src/toolkit/operating_log.cc",
    "src/toolkit/shell_agent.cc",
    "src/toolkit/string_agent.cc",
    "src/toolkit/semaphore.cc",
    "src/toolkit/managed_thread.cc",
    "src/toolkit/thread_group.cc",
    "src/toolkit/data_reader.cc",
  ],
  deps = [
    "@com_google_absl//absl/time:time",
    "@com_google_absl//absl/strings:strings",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/synchronization:synchronization",
    "@com_github_brpc_brpc//:butil",
    "@com_github_brpc_brpc//:brpc",
    "@openmpi//:openmpi",
    "@yaml_cpp//:yaml-cpp",
    "@com_google_absl//absl/hash:hash",
    ":ps_cc_proto",
    ":message",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  linkopts = [
    "-lutil",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "runtime",
  srcs = [
    "include/runtime/config_manager.h",
    "src/runtime/config_manager.cc",
  ],
  deps = [
    "@com_github_brpc_brpc//:butil",
    ":message",
    ":toolkit",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  visibility = ["//visibility:public"],
)

cc_library(
  name = "param_table",
  srcs = [
    "include/param_table/data/dense_value_ver1.h",
    "include/param_table/data/summary_value_ver1.h",
    "include/param_table/data/sparse_kv_ver1.h",
    "include/param_table/data/sparse_embedding_ver1.h",
    "include/param_table/dense_value_ver1_table.h",
    "include/param_table/summary_value_ver1_table.h",
    "include/param_table/sparse_kv_ver1_table.h",
    "include/param_table/sparse_embedding_ver1_table.h",
    "src/param_table/data/dense_value_ver1.cc",
    "src/param_table/data/summary_value_ver1.cc",
    "src/param_table/data/sparse_kv_ver1.cc",
    "src/param_table/data/sparse_embedding_ver1.cc",
    "src/param_table/dense_value_ver1_table.cc",
    "src/param_table/summary_value_ver1_table.cc",
    "src/param_table/sparse_kv_ver1_table.cc",
    "src/param_table/sparse_embedding_ver1_table.cc",
  ],
  deps = [
    "@com_google_absl//absl/synchronization:synchronization",
    "@com_google_absl//absl/random:random",
    "@com_google_absl//absl/hash:hash",
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/strings:str_format",
    "@com_github_brpc_brpc//:butil",
    ":message",
    ":toolkit",
    ":runtime",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  visibility = ["//visibility:public"],
)

cc_library(
  name = "layer",
  srcs = [
    "include/model/tool/eigen_impl.h",
    "include/model/data/component.h",
    "include/model/data/matrix_output.h",
    "include/model/layer/common/layer.h",
    "include/model/layer/common/activation_function.h",
    "src/model/layer/common/activation_function.cc",

    "include/model/layer/activation_layer.h",
    "include/model/layer/add_layer.h",
    "include/model/layer/batchnorm_layer.h",
    "include/model/layer/calibration_layer.h",
    "include/model/layer/colconcat_layer.h",
    "include/model/layer/rowconcat_layer.h",
    "include/model/layer/colrep_layer.h",
    "include/model/layer/colselect_layer.h",
    "include/model/layer/embedding_sum_layer.h",
    "include/model/layer/fusion_layer.h",
    "include/model/layer/gaussion_prob_layer.h",
    "include/model/layer/linear_layer.h",
    "include/model/layer/mul_layer.h",
    "include/model/layer/neural_layer.h",
    "include/model/layer/norm_layer.h",
    "include/model/layer/out_product_layer.h",
    "include/model/layer/product_layer.h",
    "include/model/layer/softmax_layer.h",
    "include/model/layer/sumup_layer.h",
    "include/model/layer/weight_sum_pool_layer.h",
    "src/model/layer/activation_layer.cc",
    "src/model/layer/add_layer.cc",
    "src/model/layer/batchnorm_layer.cc",
    "src/model/layer/calibration_layer.cc",
    "src/model/layer/colconcat_layer.cc",
    "src/model/layer/rowconcat_layer.cc",
    "src/model/layer/colrep_layer.cc",
    "src/model/layer/colselect_layer.cc",
    "src/model/layer/embedding_sum_layer.cc",
    "src/model/layer/fusion_layer.cc",
    "src/model/layer/gaussion_prob_layer.cc",
    "src/model/layer/linear_layer.cc",
    "src/model/layer/mul_layer.cc",
    "src/model/layer/neural_layer.cc",
    "src/model/layer/norm_layer.cc",
    "src/model/layer/out_product_layer.cc",
    "src/model/layer/product_layer.cc",
    "src/model/layer/softmax_layer.cc",
    "src/model/layer/sumup_layer.cc",
    "src/model/layer/weight_sum_pool_layer.cc",

    "include/model/layers.h",
    "src/model/layers.cc",
  ],
  deps = [
    "@com_github_brpc_brpc//:butil",
    "@eigen//:eigen",
    ":toolkit",
    ":runtime",
  ],
  includes = [
    "include",
  ],
  copts = [
    "--std=c++11",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
    "-DEIGEN_USE_MKL_ALL",
    "-fPIC",
    "-fpermissive",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "plugin",
  srcs = [
    "include/model/data/slot_array.h",
    "include/model/data/instance.h",
    "include/model/data/thread_local_data.h",

    "include/model/plugin/common/plugin.h",
    "include/model/plugin/cvm_plugin.h",
    "include/model/plugin/lr_plugin.h",
    "include/model/plugin/wide_plugin.h",
    "include/model/plugin/mf_plugin.h",
    "include/model/plugin/fm_plugin.h",
    "include/model/plugin/ps_dnn_plugin.h",
    "src/model/plugin/cvm_plugin.cc",
    "src/model/plugin/lr_plugin.cc",
    "src/model/plugin/wide_plugin.cc",
    "src/model/plugin/mf_plugin.cc",
    "src/model/plugin/fm_plugin.cc",
    "src/model/plugin/dnn_plugin.cc",

    "include/model/plugins.h",
  ],
  deps = [
    "@com_github_brpc_brpc//:butil",
    "@eigen//:eigen",
    ":toolkit",
    ":runtime",
    ":param_table",
    ":layer",
  ],
  includes = [
    "include",
  ],
  copts = [
    "--std=c++11",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
    "-DEIGEN_USE_MKL_ALL",
    "-fPIC",
    "-fpermissive",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "learner",
  srcs = [
    "include/model/tool/auc_calculator.h",
    "src/model/tool/auc_calculator.cc",
    "include/model/data/record.h",
    "src/model/data/record.cc",

    "include/model/distributed_learner/rtsparse_learner.h",
    "src/model/distributed_learner/rtsparse_learner.cc",
  ],
  deps = [
    "@com_google_absl//absl/time:time",
    "@com_google_absl//absl/strings:strings",
    "@com_github_brpc_brpc//:butil",
    ":toolkit",
    ":runtime",
    ":param_table",
    ":plugin",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  visibility = ["//visibility:public"],
)

cc_library(
  name = "runner",
  srcs = [
    "include/model/distributed_runner/rtsparse_offline_runner.h",
    "src/model/distributed_runner/rtsparse_offline_runner.cc",
    "include/model/tool/data_downloader.h",
    "src/model/tool/data_downloader.cc",
  ],
  deps = [
    "@com_google_absl//absl/strings:strings",
    "@com_google_absl//absl/strings:str_format",
    "@com_github_brpc_brpc//:butil",
    ":toolkit",
    ":runtime",
    ":learner",
  ],
  includes = [
    "include",
  ],
  copts = COPTS,
  visibility = ["//visibility:public"],
)

cc_test(
  name = "test_config_manager",
  srcs = [
    "test/runtime/test_config_manager.cc",
  ],
  deps = [
    "@com_google_googletest//:gtest",
    ":runtime",
  ],
  copts = COPTS,
  linkopts = [
    "-lgomp",
  ],
  data = glob([
    "test/runtime/data/**",
  ]),
  malloc = "@jemalloc//:jemalloc",
)

cc_test(
  name = "test_record",
  srcs = [
    "test/model/data/test_record.cc",
  ],
  deps = [
    "@com_google_googletest//:gtest",
    ":learner",
  ],
  copts = COPTS,
  linkopts = [
    "-lgomp",
  ],
  data = glob([
    "test/model/data/data/**",
  ]),
  malloc = "@jemalloc//:jemalloc",
)

cc_test(
  name = "test_sparse_kv_ver1_serialization",
  srcs = [
    "test/param_table/test_sparse_kv_ver1_serialization.cc",
  ],
  deps = [
    "@com_google_googletest//:gtest",
    ":toolkit",
    ":param_table",
  ],
  copts = COPTS,
  linkopts = [
    "-lgomp",
  ],
  data = glob([
  ]),
  malloc = "@jemalloc//:jemalloc",
)

cc_test(
  name = "test_sparse_embedding_ver1_serialization",
  srcs = [
    "test/param_table/test_sparse_embedding_ver1_serialization.cc",
  ],
  deps = [
    "@com_google_googletest//:gtest",
    ":toolkit",
    ":param_table",
  ],
  copts = COPTS,
  linkopts = [
    "-lgomp",
  ],
  data = glob([
  ]),
  malloc = "@jemalloc//:jemalloc",
)

