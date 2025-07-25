// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "buffer.h"
#include "hash_map.h"
#include "simple_vector.h"
#include "vector.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace moodist {

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
  std::visit([&](auto& v2) { x(v2); }, v);
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
  std::apply([&x](const std::decay_t<T>&... v) { x(v...); }, v);
}

template<typename X, typename... T>
void serialize(X& x, std::tuple<T...>& v) {
  std::apply([&x](std::decay_t<T>&... v) { x(v...); }, v);
}

template<typename X, typename Vector>
void serializeVector(X& x, const Vector& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}

template<typename X, typename Vector>
void serializeVector(X& x, Vector& v) {
  using T = typename Vector::value_type;
  if constexpr (std::is_trivial_v<T> && !has_serialize_helper<X>::template has_serialize<T>) {
    std::basic_string_view<T> view;
    x(view);
    v.resize(view.size());
    std::memcpy(v.data(), view.data(), sizeof(T) * view.size());
  } else {
    size_t n = x.template read<size_t>();
    v.resize(n);
    for (size_t i = 0; i != n; ++i) {
      x(v[i]);
    }
  }
}

template<typename X, typename T>
void serialize(X& x, const std::vector<T>& v) {
  serializeVector(x, v);
}

template<typename X, typename T>
void serialize(X& x, std::vector<T>& v) {
  serializeVector(x, v);
}

template<typename X, typename T, typename A>
void serialize(X& x, const Vector<T, A>& v) {
  serializeVector(x, v);
}

template<typename X, typename T, typename A>
void serialize(X& x, Vector<T, A>& v) {
  serializeVector(x, v);
}

template<typename X, typename T>
void serialize(X& x, const SimpleVector<T>& v) {
  serializeVector(x, v);
}

template<typename X, typename T>
void serialize(X& x, SimpleVector<T>& v) {
  serializeVector(x, v);
}

template<typename X, typename Map>
void serializeMap(X& x, const Map& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}

template<typename X, typename Map>
void serializeMap(X& x, Map& v) {
  v.clear();
  size_t n = x.template read<size_t>();
  for (; n; --n) {
    auto k = x.template read<typename Map::key_type>();
    v.emplace(std::move(k), x.template read<typename Map::mapped_type>());
  }
}

template<typename X, typename Key, typename Value>
void serialize(X& x, const std::map<Key, Value>& v) {
  serializeMap(x, v);
}

template<typename X, typename Key, typename Value>
void serialize(X& x, std::map<Key, Value>& v) {
  serializeMap(x, v);
}

template<typename X, typename Key, typename Value>
void serialize(X& x, const std::unordered_map<Key, Value>& v) {
  serializeMap(x, v);
}

template<typename X, typename Key, typename Value>
void serialize(X& x, std::unordered_map<Key, Value>& v) {
  serializeMap(x, v);
}

template<typename X, typename Key, typename Value>
void serialize(X& x, const HashMap<Key, Value>& v) {
  serializeMap(x, v);
}

template<typename X, typename Key, typename Value>
void serialize(X& x, HashMap<Key, Value>& v) {
  serializeMap(x, v);
}

struct OpSize {};
struct OpWrite {};
struct OpRead {};

// This is not a cross platform serializer
struct Serializer {
  std::byte* write(OpSize, std::byte* dst, [[maybe_unused]] const void* src, size_t len) {
    return (std::byte*)((uintptr_t)dst + len);
  }
  std::byte* write(OpWrite, std::byte* dst, const void* src, size_t len) {
    std::memcpy(dst, src, len);
    return dst + len;
  }
  template<typename Op, typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  std::byte* write(Op, std::byte* dst, T v) {
    dst = write(Op{}, dst, (void*)&v, sizeof(v));
    return dst;
  }
  template<typename Op>
  std::byte* write(Op, std::byte* dst, std::string_view str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), str.size());
    return dst;
  }
  template<typename Op, typename T>
  std::byte* write(Op, std::byte* dst, std::basic_string_view<T> str) {
    dst = write(Op{}, dst, str.size());
    dst = write(Op{}, dst, str.data(), sizeof(T) * str.size());
    return dst;
  }
};
struct Deserializer {
  std::string_view buf;
  Deserializer() = default;
  Deserializer(std::string_view buf) : buf(buf) {}
  Deserializer(const void* data, size_t len) : buf((const char*)data, len) {}
  [[noreturn]] void eod() {
    throw SerializationError("Deserializer: reached end of data");
  }
  void consume(size_t len) {
    buf = {buf.data() + len, buf.size() - len};
  }
  template<typename T>
  std::basic_string_view<T> readStringView() {
    size_t len = read<size_t>();
    if (buf.size() < sizeof(T) * len) {
      eod();
    }
    T* data = (T*)buf.data();
    consume(sizeof(T) * len);
    return {data, len};
  }
  std::string_view readString() {
    size_t len = read<size_t>();
    if (buf.size() < len) {
      eod();
    }
    const char* data = buf.data();
    consume(len);
    return {data, len};
  }
  template<typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void read(T& r) {
    if (buf.size() < sizeof(T)) {
      eod();
    }
    std::memcpy(&r, buf.data(), sizeof(T));
    consume(sizeof(T));
  }
  void read(std::string_view& r) {
    r = readString();
  }
  void read(std::string& r) {
    r = readString();
  }
  template<typename T>
  void read(std::basic_string_view<T>& r) {
    r = readStringView<T>();
  }

  template<typename T>
  T read() {
    T r;
    read(r);
    return r;
  }
  std::string_view read() {
    return readString();
  }

  bool empty() {
    return buf.empty();
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
  template<
      typename T,
      typename = decltype(std::declval<Serializer>().write(OpWrite{}, (std::byte*)nullptr, std::declval<T>()))>
  static std::true_type has_builtin_write_f(int);
  template<typename T>
  static const bool has_builtin_write = decltype(Serialize::has_builtin_write_f<T>(0))::value;
  template<typename T>
  void operator()(const T& v) {
    if constexpr (has_serialize<const T>) {
      v.serialize(*this);
    } else if constexpr (has_serialize<T>) {
      const_cast<T&>(v).serialize(*this);
    } else if constexpr (has_builtin_write<const T>) {
      dst = Serializer{}.write(Op{}, dst, v);
    } else {
      serialize(*this, v);
    }
  }
  template<typename... T>
  void operator()(const T&... v) {
    ((*this)(std::forward<const T&>(v)), ...);
  }

  void write(const void* data, size_t len) {
    dst = Serializer{}.write(Op{}, dst, (std::byte*)data, len);
  }

  size_t tell() const {
    return dst - begin;
  }
};

struct Deserialize {
  Deserialize(Deserializer& des) : des(des) {}
  Deserializer& des;

  template<typename T>
  static std::false_type has_serialize_f(...);
  template<typename T, typename = decltype(std::declval<T>().serialize(std::declval<Deserialize&>()))>
  static std::true_type has_serialize_f(int);
  template<typename T>
  static const bool has_serialize = decltype(Deserialize::has_serialize_f<T>(0))::value;
  template<typename T>
  static std::false_type has_builtin_read_f(...);
  template<typename T, typename = decltype(std::declval<Deserializer>().read(std::declval<T&>()))>
  static std::true_type has_builtin_read_f(int);
  template<typename T>
  static const bool has_builtin_read = decltype(Deserialize::has_builtin_read_f<T>(0))::value;
  template<typename T>
  void operator()(T& v) {
    if constexpr (has_serialize<T>) {
      v.serialize(*this);
    } else if constexpr (has_builtin_read<T>) {
      des.read(v);
    } else {
      serialize(*this, v);
    }
  }

  template<typename... T>
  void operator()(T&... v) {
    ((*this)(v), ...);
  }

  const char* consume(size_t n) {
    const char* r = des.buf.data();
    des.consume(n);
    return r;
  }

  template<typename T>
  T read() {
    if constexpr (has_serialize<T>) {
      T r;
      r.serialize(*this);
      return r;
    } else if constexpr (has_builtin_read<T>) {
      return des.read<T>();
    } else {
      T r;
      serialize(*this, r);
      return r;
    }
  }
};

template<typename Output, typename... T>
void serializeTo(Output& output, const T&... v) {
  static_assert(sizeof(*output.data()) == sizeof(std::byte));
  Serialize<OpSize> x{};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  output.resize(size);
  std::byte* dst = (std::byte*)output.data();
  Serialize<OpWrite> x2{dst, dst};
  (x2(v), ...);
}

template<typename... T>
[[gnu::warn_unused_result]] BufferHandle serializeToBuffer(const T&... v) {
  Serialize<OpSize> x{};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  BufferHandle h = makeBuffer(size);
  std::byte* dst = h->data();
  Serialize<OpWrite> x2{dst, dst};
  (x2(v), ...);
  return h;
}

template<typename... T>
void serializeToStringView(std::string_view buffer, const T&... v) {
  Serialize<OpSize> x{};
  (x(v), ...);
  size_t size = x.dst - (std::byte*)nullptr;
  if (buffer.size() < size) {
    throw SerializationError("Data does not fit in target buffer");
  }
  std::byte* dst = (std::byte*)buffer.data();
  Serialize<OpWrite> x2{dst, dst};
  (x2(v), ...);
}

template<typename... T>
size_t serializeToUnchecked(void* ptr, const T&... v) {
  std::byte* dst = (std::byte*)ptr;
  Serialize<OpWrite> x{dst, dst};
  (x(v), ...);
  return x.dst - (std::byte*)ptr;
}

template<typename... T>
std::string_view deserializeBufferPart(const void* ptr, size_t len, T&... result) {
  Deserializer des(std::string_view{(const char*)ptr, len});
  Deserialize x(des);
  x(result...);
  return des.buf;
}

template<typename... T>
std::string_view deserializeBufferPart(std::string_view data, T&... result) {
  return deserializeBufferPart(data.data(), data.size(), result...);
}

template<typename... T>
void deserializeBuffer(const void* ptr, size_t len, T&... result) {
  Deserializer des(std::string_view{(const char*)ptr, len});
  Deserialize x(des);
  x(result...);
  if (des.buf.size() != 0) {
    throw SerializationError("deserializeBuffer: " + std::to_string(des.buf.size()) + " trailing bytes");
  }
}
template<typename... T>
void deserializeBuffer(std::string_view data, T&... result) {
  deserializeBuffer(data.data(), data.size(), result...);
}
template<typename... T>
void deserializeBuffer(Buffer* buffer, T&... result) {
  Deserializer des(std::string_view{(const char*)buffer->data(), buffer->size()});
  Deserialize x(des);
  x(result...);
  if (des.buf.size() != 0) {
    throw SerializationError("deserializeBuffer: " + std::to_string(des.buf.size()) + " trailing bytes");
  }
}

template<typename... T>
std::string_view deserializeBufferPart(Buffer* buffer, T&... result) {
  Deserializer des(std::string_view{(const char*)buffer->data(), buffer->size()});
  Deserialize x(des);
  x(result...);
  return des.buf;
}

template<typename T>
struct SerializeFunction {
  const T& f;
  SerializeFunction(const T& f) : f(f) {}
  template<typename X>
  void serialize(X& x) {
    f(x);
  }
};

} // namespace moodist
