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
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace moodist {

// Helper to prevent implicit conversions in SFINAE checks for free serialize functions.
// For class types: inherits from T so template argument deduction still works.
// For non-class types: wraps value with explicit conversion only to T.
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
  if constexpr (std::is_trivially_copyable_v<T> && !has_serialize_helper<X>::template has_serialize<T>) {
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
    CHECK(checked);
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
    // dst = Serializer{}.write(Op{}, dst, (std::byte*)data, len);
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

template<typename Output, typename... T>
[[gnu::always_inline]] inline void serializeTo(Output& output, const T&... v) {
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
[[gnu::always_inline]] [[gnu::warn_unused_result]] inline BufferHandle serializeToBuffer(const T&... v) {
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
[[gnu::always_inline]] [[gnu::warn_unused_result]] inline BufferHandle serializeToBufferOneShot(const T&... v) {
  SerializeExpandable x;
  x.buffer = makeBuffer(0x80 - sizeof(Buffer));
  (x(v), ...);
  x.buffer->msize = x.size;
  return std::move(x.buffer);
}

template<typename... T>
[[gnu::always_inline]] inline void serializeToStringView(std::string_view buffer, const T&... v) {
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
[[gnu::always_inline]] inline size_t serializeToUnchecked(void* ptr, const T&... v) {
  std::byte* dst = (std::byte*)ptr;
  Serialize<OpWrite> x{dst, dst};
  (x(v), ...);
  return x.dst - (std::byte*)ptr;
}

template<typename... T>
[[gnu::always_inline]] inline std::string_view deserializeBufferPart(const void* ptr, size_t len, T&... result) {
  Deserialize x(std::string_view{(const char*)ptr, len});
  x(result...);
  return x.des.buf();
}

template<typename... T>
[[gnu::always_inline]] inline std::string_view deserializeBufferPart(std::string_view data, T&... result) {
  return deserializeBufferPart(data.data(), data.size(), result...);
}

template<typename... T>
[[gnu::always_inline]] inline void deserializeBuffer(const void* ptr, size_t len, T&... result) {
  Deserialize x(std::string_view{(const char*)ptr, len});
  x(result...);
  if (x.des.buf().size() != 0) {
    throw SerializationError("deserializeBuffer: " + std::to_string(x.des.buf().size()) + " trailing bytes");
  }
}
template<typename... T>
[[gnu::always_inline]] inline void deserializeBuffer(std::string_view data, T&... result) {
  deserializeBuffer(data.data(), data.size(), result...);
}
template<typename... T>
[[gnu::always_inline]] inline void deserializeBuffer(Buffer* buffer, T&... result) {
  Deserialize x(std::string_view{(const char*)buffer->data(), buffer->size()});
  x(result...);
  if (x.des.buf().size() != 0) {
    throw SerializationError("deserializeBuffer: " + std::to_string(x.des.buf().size()) + " trailing bytes");
  }
}

template<typename... T>
[[gnu::always_inline]] inline std::string_view deserializeBufferPart(Buffer* buffer, T&... result) {
  Deserialize x(std::string_view{(const char*)buffer->data(), buffer->size()});
  x(result...);
  return x.des.buf();
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
