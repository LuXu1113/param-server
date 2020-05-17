cc_library(
  name = "service_impl",
  srcs = [
    "include/service_impl.h",
    "src/service_impl.cc",
  ],
  includes = [
    "include",
  ],
  deps = [
    "@com_google_absl//absl/time:time",
    "@com_github_brpc_brpc//:brpc",
    "//utils:ps_cc_proto",
    "//utils:message",
    "//utils:toolkit",
    "//utils:runtime",
    "//utils:param_table",
    "//utils:runner",
  ],
  copts = [
    "--std=c++11",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
    "-DEIGEN_USE_MKL_ALL",
    "-fPIC",
  ],
  visibility = ["//visibility:public"],
)

cc_binary(
  name = "param-server",
  srcs = [
    "src/param_server.cc",
  ],
  deps = [
    "@com_google_absl//absl/strings:str_format",
    "@com_github_brpc_brpc//:brpc",
    "//utils:ps_cc_proto",
    ":service_impl",
  ],
  copts = [
    "--std=c++11",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
    "-DEIGEN_USE_MKL_ALL",
    "-fPIC",
  ],
  linkopts = [
    "-lgomp",
  ],
  # malloc = "@com_google_tcmalloc//tcmalloc:tcmalloc"
  malloc = "@jemalloc//:jemalloc"
)

cc_binary(
  name = "worker",
  srcs = [
    "src/param_server.cc",
  ],
  deps = [
    "@com_google_absl//absl/strings:str_format",
    "@com_github_brpc_brpc//:brpc",
    "//utils:ps_cc_proto",
    ":service_impl",
  ],
  copts = [
    "--std=c++11",
    "-Wno-unused-parameter",
    "-fno-omit-frame-pointer",
    "-DEIGEN_USE_MKL_ALL",
    "-fPIC",
  ],
  linkopts = [
    "-lgomp",
  ],
  # malloc = "@com_google_tcmalloc//tcmalloc:tcmalloc"
  malloc = "@jemalloc//:jemalloc"
)

