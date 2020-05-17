#ifndef UTILS_INCLUDE_TOOLKIT_CONFIG_H_
#define UTILS_INCLUDE_TOOLKIT_CONFIG_H_

#include <string>
#include <butil/logging.h>
#include <yaml-cpp/yaml.h>

namespace ps {
namespace toolkit {

// Calling operator[] to non-const YAML::Node is not thread-safe, since it will create a temporary child node when the key doesn't exist
// Calling operator[] to const YAML::Node is thread-safe, but you cannot create child node in this way
// operator[] of Config always call operator[] to const YAML::Node, so it is thread-safe to call operator[] to Config, though you cannot create child node in this way
// I also wonder how yaml-cpp manages its resource since it allows loops
// Looking for better c++ yaml library

/*
 * 封装 yaml 解析功能
 */
class Config {
 public:
  Config();
  Config(const Config& other);
  Config(const YAML::Node& node, const std::string& path);

  void load_file(const std::string& file);
  const YAML::Node& node() const;
  YAML::Node& node();

  const std::string& path() const;
  std::string& path();
  const YAML::Node& operator*() const;
  YAML::Node& operator*();
  bool is_defined() const;
  bool is_null() const;
  bool is_scalar() const;
  bool is_sequence() const;
  bool is_map() const;
  size_t size() const;

  Config& operator=(const Config& other) {
    path_ = other.path_;
    node_.reset(other.node_);
    return *this;
  }
  const YAML::Node* operator->() const {
    return &node_;
  }
  YAML::Node* operator->() {
    return &node_;
  }
  Config operator[](size_t i) const {
    return {node_[i], path_ + "[" + std::to_string(i) + "]"};
  }
  Config operator[](const std::string& key) const {
    return {node_[key], path_ + "." + key};
  }

  template<class T>
  T as() const {
    try {
      return node_.as<T>();
    } catch (const std::exception& e) {
      LOG(FATAL) << "Error reading conf item " << path_ << " : " << e.what();
      throw;
    }
  }

 private:
  YAML::Node node_;
  std::string path_;
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_CONFIG_H_

