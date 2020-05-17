#ifndef UTILS_INCLUDE_TOOLKIT_FACTORY_H_
#define UTILS_INCLUDE_TOOLKIT_FACTORY_H_

#include <memory>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <butil/logging.h>

namespace ps {
namespace toolkit {

template<class T>
class Factory {
 public:
  typedef std::function<std::shared_ptr<T>()> producer_t;

  Factory() = default;
  Factory(const Factory&) = delete;
  ~Factory() = default;

  template<class TT>
  void add(const std::string& name) {
    add(name, []() -> std::shared_ptr<TT> {
      return std::make_shared<TT>();
    });
  }
  void add(const std::string& name, producer_t producer) {
    CHECK(items_.insert({name, producer}).second) << "Factory item <" << name << "> exists already";
  }

  template<class TT = T>
  std::shared_ptr<TT> produce(const std::string& name) {
    auto it = items_.find(name);
    CHECK(it != items_.end()) << "Factory item not found: " << name;

    std::shared_ptr<T> obj = it->second();
    CHECK(obj) << "Factor item is empty: " << name;

    std::shared_ptr<TT> x = std::dynamic_pointer_cast<TT>(obj);
    CHECK(x) << "Factory item <" << name << "> can not cast from "
             << typeid(*obj).name() << " to " << typeid(TT).name();

    return x;
  }

 private:
  std::map<std::string, producer_t> items_;
};

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_FACTORY_H_

