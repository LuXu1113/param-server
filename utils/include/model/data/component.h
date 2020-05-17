#ifndef UTILS_INCLUDE_MODEL_DATA_COMPONENT_H_
#define UTILS_INCLUDE_MODEL_DATA_COMPONENT_H_

#include <type_traits>
#include <string>
#include <map>
#include <memory>
#include <butil/logging.h>
#include "toolkit/config.h"
#include "toolkit/factory.h"

namespace ps {
namespace model {

class ComponentTable;

// 组件基类，所有的神经网络对象都是component，所有的component都会注册到table中，便于搭建网络
class Component : public std::enable_shared_from_this<Component> {
 public:
  Component() = default;
  Component(const Component&) = delete;
  ~Component() = default;

  ComponentTable*& component_table() {
    return comp_table_;
  }
  virtual void load_config(ps::toolkit::Config conf) {}

  template<class T = Component>
  std::shared_ptr<T> shared_from_this() {
    std::shared_ptr<T> res = std::dynamic_pointer_cast<T>(std::enable_shared_from_this<Component>::shared_from_this());
    CHECK(res);
    return std::move(res);
  }

 private:
  ComponentTable *comp_table_ = NULL;
};

// 定义 全局的component factory
inline ps::toolkit::Factory<Component>& global_component_factory() {
  static ps::toolkit::Factory<Component> f;
  return f;
}

// 组件表的实现, 所有组件放到组件表，并用name查找
class ComponentTable {
 public:
  ComponentTable() = default;
  ComponentTable(const ComponentTable&) = delete;
  ~ComponentTable() = default;

  std::map<std::string, std::shared_ptr<Component> >& table() {
    return table_;
  }

  inline void add_component(const std::string& name, std::shared_ptr<Component> comp) {
    CHECK(name != "");
    CHECK(comp);
    CHECK(table_.insert({name, comp}).second);
  }

  template<class T>
  std::shared_ptr<T> get_component(const std::string& name) {
    if (name == "") {
      return NULL;
    }
    auto it = table_.find(name);
    CHECK(it != table_.end()) << "Component " << name << " not found.";

    std::shared_ptr<T> comp = std::dynamic_pointer_cast<T>(it->second);
    CHECK(comp) << "Cannot cast component " << name << " to type " << typeid(T).name();

    return comp;
  }

  // 加载一个组件，并加到table, 调用load config继续加载依赖组件
  // 组件定义必须按照依赖顺序
  template<class T>
  std::shared_ptr<T> load_component(ps::toolkit::Config conf) {
    LOG(INFO) << "load_component " << conf.path();
    CHECK(conf.is_scalar() || conf.is_map()) << conf.path();

    if (conf.is_scalar()) {
      LOG(INFO) << "component exist, get_component "<<conf.as<std::string>();
      return get_component<T>(conf.as<std::string>());
    }

    std::shared_ptr<T> comp;
    if (conf["class"].is_defined() && conf["class"].as<std::string>() != "") {
      LOG(INFO) << "load_component: using factory to produce "<< conf["class"].as<std::string>();
      comp = global_component_factory().produce<T>(conf["class"].as<std::string>());
    } else {
      const bool constructible = std::is_default_constructible<T>::value;
      LOG(INFO) << "load_component: using default_construct to create "<<typeid(T).name()
                <<" constructible: " << constructible;
      comp = default_construct<T>(std::integral_constant<bool, constructible>());
    }

    if (conf["name"].is_defined() && conf["name"].as<std::string>() != "") {
      LOG(INFO) << "add_component " << conf["name"].as<std::string>();
      add_component(conf["name"].as<std::string>(), comp);
    }

    comp->component_table() = this;
    comp->load_config(conf);

    return comp;
  }

  template<class T>
  std::vector<std::shared_ptr<T> > load_components(ps::toolkit::Config conf) {
    LOG(INFO) << "load_components " << conf.path();
    CHECK(conf.is_sequence()) << conf.path();

    std::vector<std::shared_ptr<T> > comps(conf.size());
    for (size_t i = 0; i < conf.size(); ++i) {
      comps[i] = load_component<T>(conf[i]);
    }
    return comps;
  }

 private:
  std::map<std::string, std::shared_ptr<Component> > table_;

  template<class T>
  std::shared_ptr<T> default_construct(std::true_type constructible) {
    return std::make_shared<T>();
  }
  template<class T>
  std::shared_ptr<T> default_construct(std::false_type constructible) {
    LOG(FATAL) << "Cannot construct component of type " << typeid(T).name();
    return NULL;
  }
};

} // namespace modps
} // namespace ps

#endif // UTILS_INCLUDE_MODEL_DATA_COMPONENT_H_

