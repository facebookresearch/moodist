// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialization utilities for serialize library - stripped down from core/serialization.h
// Removes dependencies on core headers (vector.h, hash_map.h, simple_vector.h)

#pragma once

#include "buffer.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace moodist {

// Helper to prevent implicit conversions in SFINAE checks for free serialize functions.
template<typename T, bool = std::is_class_v<T>>
struct ExactType;

template<typename T>
struct ExactType<T, true> : T {
  using T::T;
};

template<typename T>
struct ExactType<T, false> {
  const T& value;
  operator const T&() const {
    return value;
  }
  template<typename U, std::enable_if_t<!std::is_same_v<U, T>>* = nullptr>
  operator U() const = delete;
};

// Mutable version for deserialization
template<typename T, bool = std::is_class_v<T>>
struct ExactTypeMut;

template<typename T>
struct ExactTypeMut<T, true> : T {
  using T::T;
};

template<typename T>
struct ExactTypeMut<T, false> {
  T& value;
  operator T&() const {
    return value;
  }
  template<typename U, std::enable_if_t<!std::is_same_v<U, T>>* = nullptr>
  operator U&() const = delete;
};

template<typename X>
struct has_serialize_helper {
  template<typename T>
  static std::false_type has_serialize_f(...);
  template<typename T, typename = decltype(std::declval<T>().serialize(std::declval<X&>()))>
  static std::true_type has_serialize_f(int);
  template<typename T>
  static const bool has_serialize = decltype(has_serialize_f<T>(0))::value;
};

struct SerializationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

template<typename X, typename A, typename B>
void serialize(X& x, const std::pair<A, B>& v) {
  x(v.first, v.second);
}

template<typename X, typename A, typename B>
void serialize(X& x, std::pair<A, B>& v) {
  x(v.first, v.second);
}

template<typename X, typename T>
void serialize(X& x, const std::optional<T>& v) {
  x(v.has_value());
  if (v.has_value()) {
    x(v.value());
  }
}

template<typename X, typename T>
void serialize(X& x, std::optional<T>& v) {
  if (x.template read<bool>()) {
    v.emplace();
    x(v.value());
  } else {
    v.reset();
  }
}

template<typename X, typename... T>
void serialize(X& x, const std::variant<T...>& v) {
  x(v.index());
  std::visit(
      [&](auto& v2) {
        x(v2);
      },
      v);
}

template<size_t I, typename X, typename Variant, typename A, typename... T>
void deserializeVariantHelper(size_t index, X& x, Variant& variant) {
  if (index == I) {
    x(variant.template emplace<I>());
  }
  if constexpr (I + 1 != std::variant_size_v<Variant>) {
    deserializeVariantHelper<I + 1, X, Variant, T...>(index, x, variant);
  }
}

template<typename X, typename... T>
void serialize(X& x, std::variant<T...>& v) {
  size_t index = x.template read<size_t>();
  deserializeVariantHelper<0, X, std::variant<T...>, T...>(index, x, v);
}

template<typename X, typename... T>
void serialize(X& x, const std::tuple<T...>& v) {
  std::apply(
      [&x](const std::decay_t<T>&... v) {
        x(v...);
      },
      v);
}

template<typename X, typename... T>
void serialize(X& x, std::tuple<T...>& v) {
  std::apply(
      [&x](std::decay_t<T>&... v) {
        x(v...);
      },
      v);
}

// Serialize std::vector
template<typename X, typename T>
void serialize(X& x, const std::vector<T>& v) {
  x(v.size());
  for (const auto& item : v) {
    x(item);
  }
}

template<typename X, typename T>
void serialize(X& x, std::vector<T>& v) {
  size_t n = x.template read<size_t>();
  v.resize(n);
  for (size_t i = 0; i != n; ++i) {
    x(v[i]);
  }
}

struct OpSize {};
struct OpWrite {};
struct OpRead {};

// This is not a cross platform serializer
struct Serializer {
  [[gnu::always_inline]] std::byte* write(OpSize, std::byte* dst, [[maybe_unused]] const void* src, size_t len) {
    return (std::byte*)((uintptr_t)dst + len);
  }
  [[gnu::always_inline]] std::byte* write(OpWrite, std::byte* dst, const void* src, size_t len) {
    std::memcpy(dst, src, len);
    return dst + len;
  }
  template<typename Op, typename T, std::enable_if_t<std::is_trivially_copyable_v<T>>* = nullptr>
  [[gnu::always_inline]] std::byte* write(Op, std::byte* dst, T v) {
    dst = write(Op{}, dst, (void*)&v, sizeof(v));
    return dst;
  }
  template<typename Op>
  [[gnu::always_inline]] std::byte* write(Op, std::byte* dst, std::string_view str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), str.size());
    return dst;
  }
  template<typename Op, typename T>
  [[gnu::always_inline]] std::byte* write(Op, std::byte* dst, std::basic_string_view<T> str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), sizeof(T) * str.size());
    return dst;
  }
};

template<bool checked>
struct Deserializer {
  const char* base = nullptr;
  size_t offset = 0;
  size_t length = 0;
  Deserializer() = default;
  Deserializer(std::string_view buf) : base(buf.data()), offset(0), length(buf.size()) {}
  Deserializer(const void* data, size_t len) : base((const char*)data), offset(0), length(len) {}
  [[noreturn]] void eod() {
    throw SerializationError("Deserializer: reached end of data");
  }
  [[gnu::always_inline]] void consume(size_t len) {
    offset += len;
  }
  [[gnu::always_inline]] std::string_view buf() {
    return {base + offset, base + length};
  }
  [[gnu::always_inline]] std::string_view buf() const {
    return {base + offset, base + length};
  }
  template<typename T>
  std::basic_string_view<T> readStringView() {
    size_t len = read<size_t>();
    if (checked && length - offset < sizeof(T) * len) [[unlikely]] {
      eod();
    }
    T* data = (T*)(base + offset);
    consume(sizeof(T) * len);
    return {data, len};
  }
  std::string_view readString() {
    size_t len = read<size_t>();
    if (checked && length - offset < len) [[unlikely]] {
      eod();
    }
    const char* data = base + offset;
    consume(len);
    return {data, len};
  }
  template<typename T, std::enable_if_t<std::is_trivially_copyable_v<T>>* = nullptr>
  [[gnu::always_inline]] void read(T& r) {
    if (checked && length - offset < sizeof(T)) [[unlikely]] {
      eod();
    }
    std::memcpy(&r, base + offset, sizeof(T));
    consume(sizeof(T));
  }
  [[gnu::always_inline]] void read(std::string_view& r) {
    r = readString();
  }
  [[gnu::always_inline]] void read(std::string& r) {
    r = readString();
  }
  template<typename T>
  [[gnu::always_inline]] void read(std::basic_string_view<T>& r) {
    r = readStringView<T>();
  }

  template<typename T>
  [[gnu::always_inline]] T read() {
    T r;
    read(r);
    return r;
  }
  [[gnu::always_inline]] std::string_view read() {
    return readString();
  }

  [[gnu::always_inline]] bool empty() {
    return offset == length;
  }
};

template<typename Op>
struct Serialize {
  std::byte* begin = nullptr;
  std::byte* dst = nullptr;
  template<typename T>
  static std::false_type has_serialize_f(...);
  template<typename T, typename = decltype(std::declval<T>().serialize(std::declval<Serialize&>()))>
  static std::true_type has_serialize_f(int);
  template<typename T>
  static const bool has_serialize = decltype(Serialize::has_serialize_f<T>(0))::value;
  template<typename T>
  static std::false_type has_builtin_write_f(...);
  template<typename T,
      typename = decltype(std::declval<Serializer>().write(OpWrite{}, (std::byte*)nullptr, std::declval<T>()))>
  static std::true_type has_builtin_write_f(int);
  template<typename T>
  static const bool has_builtin_write = decltype(Serialize::has_builtin_write_f<T>(0))::value;
  template<typename T>
  static std::false_type has_free_serialize_f(...);
  template<typename T, typename = decltype(serialize(std::declval<Serialize&>(), std::declval<ExactType<T>>()))>
  static std::true_type has_free_serialize_f(int);
  template<typename T>
  static const bool has_free_serialize = decltype(Serialize::has_free_serialize_f<T>(0))::value;
  template<typename T>
  [[gnu::always_inline]] void operator()(const T& v) {
    if constexpr (has_serialize<const T>) {
      v.serialize(*this);
    } else if constexpr (has_serialize<T>) {
      const_cast<T&>(v).serialize(*this);
    } else if constexpr (has_free_serialize<T>) {
      serialize(*this, v);
    } else if constexpr (has_builtin_write<const T>) {
      dst = Serializer{}.write(Op{}, dst, v);
    } else {
      static_assert(false, "No serialization defined for type");
    }
  }
  template<typename... T>
  [[gnu::always_inline]] void operator()(const T&... v) {
    ((*this)(std::forward<const T&>(v)), ...);
  }

  [[gnu::always_inline]] void write(const void* data, size_t len) {
    dst = Serializer{}.write(Op{}, dst, (std::byte*)data, len);
  }

  size_t tell() const {
    return dst - begin;
  }
};

struct SerializeExpandable {
  BufferHandle buffer;
  size_t size = 0;
  template<typename T>
  static std::false_type has_serialize_f(...);
  template<typename T, typename = decltype(std::declval<T>().serialize(std::declval<SerializeExpandable&>()))>
  static std::true_type has_serialize_f(int);
  template<typename T>
  static const bool has_serialize = decltype(SerializeExpandable::has_serialize_f<T>(0))::value;
  template<typename T>
  static std::false_type has_builtin_write_f(...);
  template<typename T,
      typename = decltype(std::declval<Serializer>().write(OpWrite{}, (std::byte*)nullptr, std::declval<T>()))>
  static std::true_type has_builtin_write_f(int);
  template<typename T>
  static const bool has_builtin_write = decltype(SerializeExpandable::has_builtin_write_f<T>(0))::value;
  template<typename T>
  static std::false_type has_free_serialize_f(...);
  template<typename T,
      typename = decltype(serialize(std::declval<SerializeExpandable&>(), std::declval<ExactType<T>>()))>
  static std::true_type has_free_serialize_f(int);
  template<typename T>
  static const bool has_free_serialize = decltype(SerializeExpandable::has_free_serialize_f<T>(0))::value;
  template<typename T>
  [[gnu::always_inline]] void operator()(const T& v) {
    if constexpr (has_serialize<const T>) {
      v.serialize(*this);
    } else if constexpr (has_serialize<T>) {
      const_cast<T&>(v).serialize(*this);
    } else if constexpr (has_free_serialize<T>) {
      serialize(*this, v);
    } else if constexpr (has_builtin_write<const T>) {
      twrite(v);
    } else {
      static_assert(false, "No serialization defined for type");
    }
  }
  template<typename... T>
  [[gnu::always_inline]] void operator()(const T&... v) {
    ((*this)(std::forward<const T&>(v)), ...);
  }

  [[gnu::always_inline]] void write(const void* data, size_t len) {
    twrite(data, len);
  }

  [[gnu::noinline]] void expand(size_t nreq) {
    BufferHandle newbuffer = makeBuffer(std::max(buffer->msize * 2, buffer->msize + nreq));
    newbuffer->msize = internalAllocSize(newbuffer) - sizeof(Buffer);
    std::memcpy(newbuffer->data(), buffer->data(), size);
    std::swap(buffer, newbuffer);
  }

  template<typename... Args>
  [[gnu::always_inline]] void twrite(Args&&... args) {
    std::byte* zero = nullptr;
    std::byte* end = Serializer{}.write(OpSize{}, zero, args...);
    size_t nreq = end - zero;
    if (nreq >= buffer->msize - size) [[unlikely]] {
      expand(nreq);
    }
    Serializer{}.write(OpWrite{}, buffer->data() + size, args...);
    size += nreq;
  }

  size_t tell() const {
    return size;
  }

  void ensure(size_t n) {
    if (buffer->msize - size < n) [[unlikely]] {
      expand(n);
    }
  }
  void* data() const {
    return buffer->data() + size;
  }
  void advance(size_t n) {
    size += n;
  }
};

template<typename T>
constexpr bool is_string_view = false;
template<typename T>
constexpr bool is_string_view<std::basic_string_view<T>> = true;

template<bool checked = true>
struct Deserialize {
  Deserializer<checked> des;

  Deserialize() = default;
  Deserialize(std::string_view buf) : des(buf) {}
  Deserialize(const void* data, size_t len) : des(data, len) {}

  template<typename T>
  static std::false_type has_serialize_f(...);
  template<typename T, typename = decltype(std::declval<T>().serialize(std::declval<Deserialize&>()))>
  static std::true_type has_serialize_f(int);
  template<typename T>
  static const bool has_serialize = decltype(Deserialize::has_serialize_f<T>(0))::value;
  template<typename T>
  static std::false_type has_builtin_read_f(...);
  template<typename T, typename = decltype(std::declval<Deserializer<checked>>().read(std::declval<T&>()))>
  static std::true_type has_builtin_read_f(int);
  template<typename T>
  static const bool has_builtin_read = decltype(Deserialize::has_builtin_read_f<T>(0))::value;
  template<typename T>
  static std::false_type has_free_serialize_f(...);
  template<typename T, typename = decltype(serialize(std::declval<Deserialize&>(), std::declval<ExactTypeMut<T>>()))>
  static std::true_type has_free_serialize_f(int);
  template<typename T>
  static const bool has_free_serialize = decltype(Deserialize::has_free_serialize_f<T>(0))::value;
  template<typename T>
  [[gnu::always_inline]] void operator()(T& v) {
    if constexpr (has_serialize<T>) {
      v.serialize(*this);
    } else if constexpr (has_free_serialize<T>) {
      serialize(*this, v);
    } else if constexpr (has_builtin_read<T>) {
      des.read(v);
    } else {
      static_assert(false, "No deserialization defined for type");
    }
  }

  template<typename T>
  static constexpr bool copy_trivially =
      std::is_trivially_copyable_v<T> && !has_serialize<T> && has_builtin_read<T> && !is_string_view<T>;

  template<int>
  constexpr static size_t numTrivials() {
    return 0;
  }
  template<int, typename A, typename... T>
  constexpr static size_t numTrivials() {
    return copy_trivially<A> ? 1 + numTrivials<0, T...>() : 0;
  }
  template<int>
  constexpr static size_t trivialBytes() {
    return 0;
  }
  template<int, typename A, typename... T>
  constexpr static size_t trivialBytes() {
    return copy_trivially<A> ? sizeof(A) + trivialBytes<0, T...>() : 0;
  }

  template<size_t index, size_t end>
  constexpr static void trivialCopy() {
    static_assert(index == end);
  }
  template<size_t index, size_t end, typename A, typename... T>
  [[gnu::always_inline]] void trivialCopy(A& a, T&... v) {
    static_assert(numTrivials<0, A, T...>() == end - index);
    if constexpr (index != end) {
      static_assert(copy_trivially<A>);
      std::memcpy(&a, des.base + des.offset, sizeof(A));
      des.offset += sizeof(A);
      trivialCopy<index + 1, end>(v...);
    } else {
      static_assert(!copy_trivially<A>);
      (*this)(a);
      if constexpr (sizeof...(T)) {
        (*this)(v...);
      }
    }
  }

  template<typename A, typename... T>
  [[gnu::always_inline]] void operator()(A& a, T&... v) {
    static_assert(sizeof...(T));
    constexpr size_t nTrivial = numTrivials<0, A, T...>();
    if constexpr (nTrivial) {
      constexpr size_t nbytes = trivialBytes<0, A, T...>();
      if (checked && des.buf().size() < nbytes) [[unlikely]] {
        des.eod();
      }
      trivialCopy<0, nTrivial, A, T...>(a, v...);
    } else {
      (*this)(a);
      (*this)(v...);
    }
  }

  const char* consume(size_t n) {
    const char* r = des.base + des.offset;
    if (checked && des.buf().size() < n) [[unlikely]] {
      des.eod();
    }
    des.consume(n);
    return r;
  }

  template<typename T>
  [[gnu::always_inline]] T read() {
    if constexpr (has_serialize<T>) {
      T r;
      r.serialize(*this);
      return r;
    } else if constexpr (has_builtin_read<T>) {
      return des.template read<T>();
    } else {
      T r;
      serialize(*this, r);
      return r;
    }
  }

  size_t remaining() const {
    return des.buf().size();
  }

  size_t tell() const {
    return des.offset;
  }

  const void* data() const {
    return des.base + des.offset;
  }
  void consumeNoCheck(size_t n) {
    des.consume(n);
  }

  Deserialize<false>& unchecked() {
    return (Deserialize<false>&)(*this);
  }
};

} // namespace moodist
