#ifndef UTILS_INCLUDE__TOOLKIT_ARCHIVE_H_
#define UTILS_INCLUDE__TOOLKIT_ARCHIVE_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <butil/logging.h>

namespace ps {
namespace toolkit {

class ArchiveBase {
 public:
  ArchiveBase();
  ArchiveBase(const ArchiveBase&) = delete;
  ArchiveBase(ArchiveBase&& other);
  ArchiveBase(const std::string& data);
  ~ArchiveBase();

  ArchiveBase& operator=(ArchiveBase&& other);

  char *buffer();
  char *cursor();
  char *finish();
  char *limit();

  size_t position();
  size_t length();
  size_t capacity();
  bool empty();

  void set_read_buffer(const std::string& data);
  void set_read_buffer(const char* buffer, size_t length);
  void set_write_buffer(const char* buffer, size_t capacity);
  void set_buffer(const char *buffer, size_t length, size_t capacity);

  void set_cursor(char *cursor);
  void advance_cursor(size_t offset);

  void set_finish(char* finish);
  void advance_finish(size_t offset);

  void reset();
  void clear();
  char *release();
  void release(std::string *str);
  void resize(const size_t newsize);
  void reserve(const size_t newcap);

  void prepare_read(size_t size);
  void prepare_write(size_t size);
  void read(void* data, size_t size);
  void read_back(void* data, size_t size);
  void write(const void* data, size_t size);

  template<class T>
  void get_raw(T& x) {
    prepare_read(sizeof(T));
    memcpy(&x, cursor_, sizeof(T));
    advance_cursor(sizeof(T));
  }
  template<class T>
  T get_raw() {
    T x;
    get_raw<T>(x);
    return x;
  }
  template<class T>
  void put_raw(const T& x) {
    prepare_write(sizeof(T));
    memcpy(finish_, &x, sizeof(T));
    advance_finish(sizeof(T));
  }

 protected:
  char *buffer_;
  char *cursor_;
  char *finish_;
  char *limit_;

  void free_buffer();
};

class BinaryArchive : public ArchiveBase {
 public:
  #define PS_REPEAT_PATTERN(T)                   \
  BinaryArchive& operator>>(T& x) {              \
    get_raw(x);                                  \
    return *this;                                \
  }                                              \
  BinaryArchive& operator<<(const T& x) {        \
    put_raw(x);                                  \
    return *this;                                \
  }
  PS_REPEAT_PATTERN(int16_t)
  PS_REPEAT_PATTERN(uint16_t)
  PS_REPEAT_PATTERN(int32_t)
  PS_REPEAT_PATTERN(uint32_t)
  PS_REPEAT_PATTERN(int64_t)
  PS_REPEAT_PATTERN(uint64_t)
  PS_REPEAT_PATTERN(float)
  PS_REPEAT_PATTERN(double)
  PS_REPEAT_PATTERN(char)
  PS_REPEAT_PATTERN(signed char)
  PS_REPEAT_PATTERN(unsigned char)
  PS_REPEAT_PATTERN(bool)
  #undef PS_REPEAT_PATTERN

  template<class T>
  T get() {
    T x;
    *this >> x;
    return x;
  }
};

template<class T>
BinaryArchive& operator<<(BinaryArchive& ar, const std::vector<T>& p) {
  ar << (size_t)p.size();
  for (const auto& x : p) {
    ar << x;
  }
  return ar;
}

template<class T>
BinaryArchive& operator>>(BinaryArchive& ar, std::vector<T>& p) {
  p.resize(ar.get<size_t>());
  for (auto& x : p) {
    ar >> x;
  }
  return ar;
}

BinaryArchive& operator<<(BinaryArchive& ar, const std::string& s);
BinaryArchive& operator>>(BinaryArchive& ar, std::string& s);

// template<class T1, class T2>
// BinaryArchive& operator<<(BinaryArchive& ar, const std::pair<T1, T2>& x) {
//   return ar << x.first << x.second;
// }
//
// template<class T1, class T2>
// BinaryArchive& operator>>(BinaryArchive& ar, std::pair<T1, T2>& x) {
//   return ar >> x.first >> x.second;
// }
//
// template<class... T>
// BinaryArchive& serialize_tuple(BinaryArchive& ar, const std::tuple<T...>& x, std::integral_constant<size_t, 0> n) {
//   return ar;
// }
//
// template<class... T, size_t N>
// BinaryArchive& serialize_tuple(BinaryArchive& ar, const std::tuple<T...>& x, std::integral_constant<size_t, N> n) {
//   return serialize_tuple(ar, x, std::integral_constant<size_t, N - 1>()) << std::get<N - 1>(x);
// }
//
// template<class... T>
// BinaryArchive& operator<<(BinaryArchive& ar, const std::tuple<T...>& x) {
//   const size_t size = std::tuple_size<std::tuple<T...>>::value;
//   return serialize_tuple(ar, x, std::integral_constant<size_t, size>());
// }
//
// template<class... T>
// BinaryArchive& deserialize_tuple(BinaryArchive& ar, std::tuple<T...>& x, std::integral_constant<size_t, 0> n) {
//   return ar;
// }
//
// template<class... T, size_t N>
// BinaryArchive& deserialize_tuple(BinaryArchive& ar, std::tuple<T...>& x, std::integral_constant<size_t, N> n) {
//   return deserialize_tuple(ar, x, std::integral_constant<size_t, N - 1>()) >> std::get<N - 1>(x);
// }
//
// template<class... T>
// BinaryArchive& operator>>(BinaryArchive& ar, std::tuple<T...>& x) {
//   const size_t size = std::tuple_size<std::tuple<T...>>::value;
//   return deserialize_tuple(ar, x, std::integral_constant<size_t, size>());
// }
//
// #define PS_REPEAT_PATTERN(MAP_TYPE, RESERVE_STATEMENT)                                   \
//   template<class KEY, class VALUE, class... ARGS>                                        \
//   BinaryArchive& operator<<(BinaryArchive& ar, const MAP_TYPE<KEY, VALUE, ARGS...>& p) { \
//     ar << (size_t)p.size();                                                              \
//     for (auto it = p.begin(); it != p.end(); ++it) {                                     \
//       ar << *it;                                                                         \
//     }                                                                                    \
//     return ar;                                                                           \
//   }                                                                                      \
//   template<class KEY, class VALUE, class... ARGS>                                        \
//   BinaryArchive& operator>>(BinaryArchive& ar, MAP_TYPE<KEY, VALUE, ARGS...>& p) {       \
//     size_t size = ar.get<size_t>();                                                      \
//     p.clear();                                                                           \
//     RESERVE_STATEMENT;                                                                   \
//     for (size_t i = 0; i < size; i++) {                                                  \
//       p.insert(ar.get<std::pair<KEY, VALUE>>());                                         \
//     }                                                                                    \
//     return ar;                                                                           \
//   }                                                                                      \
//
// PS_REPEAT_PATTERN(std::map, )
// PS_REPEAT_PATTERN(std::multimap, )
// PS_REPEAT_PATTERN(std::unordered_map, p.reserve(size))
// PS_REPEAT_PATTERN(std::unordered_multimap, p.reserve(size))
// #undef PS_REPEAT_PATTERN
//
// #define PS_REPEAT_PATTERN(SET_TYPE, RESERVE_STATEMENT)                                   \
//   template<class KEY, class... ARGS>                                                     \
//   BinaryArchive& operator<<(BinaryArchive& ar, const SET_TYPE<KEY, ARGS...>& p) {        \
//     ar << (size_t)p.size();                                                              \
//     for (auto it = p.begin(); it != p.end(); ++it) {                                     \
//       ar << *it;                                                                         \
//     }                                                                                    \
//     return ar;                                                                           \
//   }                                                                                      \
//   template<class KEY, class... ARGS>                                                     \
//   BinaryArchive& operator>>(BinaryArchive& ar, SET_TYPE<KEY, ARGS...>& p) {              \
//     size_t size = ar.get<size_t>();                                                      \
//     p.clear();                                                                           \
//     RESERVE_STATEMENT;                                                                   \
//     for (size_t i = 0; i < size; i++) {                                                  \
//       p.insert(ar.get<KEY>());                                                           \
//     }                                                                                    \
//     return ar;                                                                           \
//   }                                                                                      \
//
// PS_REPEAT_PATTERN(std::set, )
// PS_REPEAT_PATTERN(std::multiset, )
// PS_REPEAT_PATTERN(std::unordered_set, p.reserve(size))
// PS_REPEAT_PATTERN(std::unordered_multiset, p.reserve(size))
// #undef PS_REPEAT_PATTERN

} // namespace toolkit
} // namespace ps

#endif // UTILS_INCLUDE_TOOLKIT_ARCHIVE_H_

