#include "toolkit/archive.h"

#include <stdlib.h>
#include <string>
#include <butil/logging.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

using std::string;

namespace ps {
namespace toolkit {

ArchiveBase::ArchiveBase() :
  buffer_(NULL),
  cursor_(NULL),
  finish_(NULL),
  limit_(NULL) {
}

ArchiveBase::ArchiveBase(ArchiveBase&& other) :
  buffer_(other.buffer_),
  cursor_(other.cursor_),
  finish_(other.finish_),
  limit_(other.limit_) {
    other.buffer_ = NULL;
    other.cursor_ = NULL;
    other.finish_ = NULL;
    other.limit_  = NULL;
}

ArchiveBase::~ArchiveBase() {
  reset();
}

ArchiveBase& ArchiveBase::operator=(ArchiveBase&& other) {
  if (this != &other) {
    reset();
    buffer_ = other.buffer_;
    cursor_ = other.cursor_;
    finish_ = other.finish_;
    limit_  = other.limit_;
    other.buffer_ = NULL;
    other.cursor_ = NULL;
    other.finish_ = NULL;
    other.limit_  = NULL;
  }
  return *this;
}

char *ArchiveBase::buffer() {
  return buffer_;
}
char *ArchiveBase::cursor() {
  return cursor_;
}
char *ArchiveBase::finish() {
  return finish_;
}
char *ArchiveBase::limit() {
  return limit_;
}

size_t ArchiveBase::position() {
  return cursor_ - buffer_;
}
size_t ArchiveBase::length() {
  return finish_ - buffer_;
}
size_t ArchiveBase::capacity() {
  return limit_ - buffer_;
}
bool ArchiveBase::empty() {
  return finish_ == buffer_;
}

void ArchiveBase::reset() {
  if (NULL != buffer_) {
    free(buffer_);
  }
  buffer_ = NULL;
  cursor_ = NULL;
  finish_ = NULL;
  limit_  = NULL;
}

void ArchiveBase::clear() {
  cursor_ = buffer_;
  finish_ = buffer_;
}

char *ArchiveBase::release() {
  char *buf = buffer_;
  buffer_ = NULL;
  cursor_ = NULL;
  finish_ = NULL;
  limit_  = NULL;
  return buf;
}

void ArchiveBase::release(string *str) {
  int len = length();
  str->clear();
  try {
    str->reserve(len);
    str->assign(buffer_, len);
  } catch (std::bad_alloc& ba) {
    LOG(FATAL) << "can not allocate memory.";
  }
  reset();
}

void ArchiveBase::resize(const size_t newsize) {
  if (unlikely(newsize > capacity())) {
    reserve(std::max(capacity() * 2, newsize));
  }
  finish_ = buffer_ + newsize;
  cursor_ = std::min(cursor_, finish_);
}

void ArchiveBase::reserve(const size_t newcap) {
  if (newcap > capacity()) {
    char *newbuf = NULL;
    newbuf = (char *)malloc(newcap);
    CHECK(NULL != newbuf) << "can not allocate memory.";
    if (length() > 0) {
      memcpy(newbuf, buffer_, length());
    }
    cursor_ = newbuf + (cursor_ - buffer_);
    finish_ = newbuf + (finish_ - buffer_);
    limit_  = newbuf + newcap;
    if (NULL != buffer_) {
      free(buffer_);
      buffer_ = NULL;
    }
    buffer_ = newbuf;
  }
}

void ArchiveBase::set_read_buffer(const string& data) {
  set_read_buffer(data.c_str(), data.length());
}

void ArchiveBase::set_read_buffer(const char *buffer, size_t length) {
  set_buffer(buffer, length, length);
}

void ArchiveBase::set_write_buffer(const char *buffer, size_t capacity) {
  set_buffer(buffer, 0, capacity);
}

void ArchiveBase::set_buffer(const char *buffer, size_t length, size_t capacity) {
  CHECK(length <= capacity) << "length = " << length << ", capacity = " << capacity;
  if (length > 0) {
    reset();
    char *new_buf = (char *)malloc(capacity);
    CHECK(NULL != new_buf) << "can not allocate memory.";
    memcpy(new_buf, buffer, length);
    buffer_ = new_buf;
    cursor_ = new_buf;
    finish_ = new_buf + length;
    limit_  = new_buf + capacity;
  }
}

void ArchiveBase::set_cursor(char *cursor) {
  CHECK(cursor >= buffer_ && cursor <= finish_)
    << "buffer = " << buffer_ << ", finish = " << finish_ << ", cursor = " << cursor;
  cursor_ = cursor;
}

void ArchiveBase::advance_cursor(size_t offset) {
  CHECK(offset <= size_t(finish_ - cursor_))
    << "finish - cursor = " << finish_ - cursor_ << ", offset = " << offset;
  cursor_ += offset;
}

void ArchiveBase::set_finish(char* finish) {
  CHECK(finish >= cursor_ && finish <= limit_)
    << "cursor = " << cursor_ << ", limit = " << limit_ << ", finish = " << finish;
  finish_ = finish;
}

void ArchiveBase::advance_finish(size_t offset) {
  CHECK(offset <= size_t(limit_ - finish_))
    << "limit - finish = " << limit_ - finish_ << ", offset = " << offset;
  finish_ += offset;
}

void ArchiveBase::prepare_read(size_t size) {
  if (unlikely(!(size <= size_t(finish_ - cursor_)))) {
    CHECK(size <= size_t(finish_ - cursor_))
      << "finish - cursor = " << finish_ - cursor_ << ", size = " << size;
  }
}

void ArchiveBase::prepare_write(size_t size) {
  if (unlikely(size > size_t(limit_ - finish_))) {
    reserve(std::max(capacity() * 2, length() + size));
  }
}

void ArchiveBase::read(void *data, size_t size) {
  if (size > 0) {
    prepare_read(size);
    memcpy(data, cursor_, size);
    advance_cursor(size);
  }
}

void ArchiveBase::read_back(void *data, size_t size) {
  if (size > 0) {
    CHECK(size <= size_t(finish_ - cursor_))
      << "finish - cursor = " << finish_ - cursor_ << ", size = " << size;
    memcpy(data, finish_ - size, size);
    finish_ -= size;
  }
}

void ArchiveBase::write(const void *data, size_t size) {
  if (size > 0) {
    prepare_write(size);
    memcpy(finish_, data, size);
    advance_finish(size);
  }
}

BinaryArchive& operator<<(BinaryArchive& ar, const std::string& s) {
  ar << (size_t)s.length();
  ar.write(&s[0], s.length());
  return ar;
}

BinaryArchive& operator>>(BinaryArchive& ar, std::string& s) {
  size_t len = ar.template get<size_t>();
  ar.prepare_read(len);
  s.assign(ar.cursor(), len);
  ar.advance_cursor(len);
  return ar;
}

} // namespace toolkit
} // namespace ps

#undef unlikely
#undef likely

