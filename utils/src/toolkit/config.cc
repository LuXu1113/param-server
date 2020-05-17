#include "toolkit/config.h"

#include <butil/logging.h>
#include <yaml-cpp/yaml.h>
#include "toolkit/string_agent.h"

namespace ps {
namespace toolkit {

Config::Config() :
  node_(YAML::NodeType::Undefined),
  path_() {
}

Config::Config(const Config& other) :
  node_(other.node_),
  path_(other.path_) {
}

Config::Config(const YAML::Node& node, const std::string& path) :
  node_(node),
  path_(path) {
}

void Config::load_file(const std::string &file) {
  node_ = YAML::LoadFile(file);
}

const YAML::Node& Config::node() const {
  return node_;
}

YAML::Node& Config::node() {
  return node_;
}

const std::string& Config::path() const {
  return path_;
}

std::string& Config::path() {
  return path_;
}

const YAML::Node& Config::operator*() const {
  return node_;
}

YAML::Node& Config::operator*() {
  return node_;
}

bool Config::is_defined() const {
  return node_.IsDefined();
}

bool Config::is_null() const {
  return node_.IsNull();
}

bool Config::is_scalar() const {
  return node_.IsScalar();
}

bool Config::is_sequence() const {
  return node_.IsSequence();
}

bool Config::is_map() const {
  return node_.IsMap();
}

size_t Config::size() const {
  return node_.size();
}

} // namespace toolkit
} // namespace ps

