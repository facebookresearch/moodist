
#include "buffer.h"
#include "pybind11/numpy.h"
#include <ATen/ops/from_blob.h>
#include <abstract.h>
#include <c10/util/python_stub.h>
#include <dictobject.h>
#include <floatobject.h>
#include <listobject.h>
#include <object.h>
#include <optional>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cpython/longintrepr.h>

#include <pyerrors.h>
#include <torch/python.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <unicodeobject.h>

#include "common.h"
#include "serialization.h"

namespace moodist {

namespace py = pybind11;

enum pyTypes : uint8_t {
  none,
  bool_true,
  bool_false,
  int8,
  int16,
  int32,
  int64,
  float_,
  dict,
  str,
  list,
  tuple,
  bigint,
  unicode,
  bytes,
  bytearray,
  set,
  frozenset,
  global,
  class_,
  offset,
};

namespace {

constexpr long intlistRange = 1024 * 1024 * 2;
std::array<PyObject*, intlistRange>& intlist = Global();
constexpr long intlistRangeBegin = -(long)(intlistRange / 2);
constexpr long intlistRangeEnd = (long)(intlistRange / 2);
std::once_flag globalsInitFlag;

PyObject* strDot;
PyObject* strSetState;
PyObject* strQualName;
PyObject* strModule;
PyObject* strReduceEx;
PyObject* strReduce;
PyObject* strDict;
PyObject* strExtend;

void globalsInit() {
  auto c = [](PyObject* o) {
    if (!o) {
      throw py::error_already_set();
    }
    return o;
  };
  auto start = std::chrono::steady_clock::now();
  for (long v = intlistRangeBegin; v < intlistRangeEnd; ++v) {
    size_t index = v + intlistRange / 2;
    intlist.at(index) = c(PyLong_FromLong(v));
  }

  strDot = c(PyUnicode_FromString("."));
  strSetState = c(PyUnicode_FromString("__setstate__"));
  strQualName = c(PyUnicode_FromString("__qualname__"));
  strModule = c(PyUnicode_FromString("__module__"));
  strReduceEx = c(PyUnicode_FromString("__reduce_ex__"));
  strReduce = c(PyUnicode_FromString("__reduce__"));
  strDict = c(PyUnicode_FromString("__dict__"));
  strExtend = c(PyUnicode_FromString("extend"));
}

struct ObjHash {
  size_t operator()(PyObject* o) {
    // return ((uintptr_t)o / 16ul);
    return ((uintptr_t)o / 16ul) % 137438953471ul;
    // return ((uintptr_t)o * 2654435761);
    //  return ((uintptr_t)o) % 137438953471ul;
  }
  size_t operator()(size_t v) {
    return v * 2654435761;
  }
};

template<typename Key, typename Value>
struct ObjMap {
  static_assert(std::is_trivial_v<Key>);
  static_assert(std::is_trivial_v<Value>);

  size_t allocated = 0;
  size_t msize = 0;

  struct Item {
    Key key;
    Value value;
  };

  Item* items = nullptr;
  size_t* indices = nullptr;

  static std::tuple<Item*, size_t*> alloc(size_t n) {
    auto* items = InternalAllocator<Item>().allocate(n);
    auto* indices = InternalAllocator<size_t>().allocate(n);
    if (!items || !indices) {
      if (items) {
        InternalAllocator<Item>().deallocate(items, n);
      }
      if (indices) {
        InternalAllocator<size_t>().deallocate(indices, n);
      }
    }
    return {items, indices};
  }

  ObjMap() {
    allocated = 1024 * 1024;
    std::tie(items, indices) = alloc(allocated);
    std::memset(items, 0, sizeof(Item) * allocated);
  }
  ObjMap(ObjMap&) = delete;
  ObjMap& operator=(ObjMap&) = delete;

  ~ObjMap() {
    InternalAllocator<Item>().deallocate(items, allocated);
    InternalAllocator<size_t>().deallocate(indices, allocated);
  }

  [[gnu::noinline]] [[gnu::cold]] void expand() {
    size_t newallocated = allocated * 2;
    auto [newitems, newindices] = alloc(newallocated);
    std::memset(newitems, 0, sizeof(Item) * newallocated);
    auto* olditems = items;
    auto* oldindices = indices;
    size_t oldmsize = msize;
    items = newitems;
    indices = newindices;
    allocated = newallocated;
    msize = 0;

    try {
      for (size_t i = 0; i != oldmsize; ++i) {
        size_t index = oldindices[i];
        add(olditems[index].key, olditems[index].value);
      }
    } catch (...) {
      InternalAllocator<size_t>().deallocate(oldindices, allocated);
      InternalAllocator<Item>().deallocate(olditems, allocated);
      throw;
    }
    InternalAllocator<size_t>().deallocate(oldindices, allocated);
    InternalAllocator<Item>().deallocate(olditems, allocated);
  }

  [[gnu::always_inline]] [[gnu::hot]] std::optional<Value> add(Key key, Value value) {
    while (true) {
      size_t index = ObjHash()(key) % (allocated - 1);
      if (!items[index].key) [[likely]] {
        indices[msize] = index;
        ++msize;
        items[index] = {key, value};
        return {};
      }
      for (size_t o = 0; o != 4; ++o) {
        if (items[index + o].key == key) {
          return items[index + o].value;
        }
      }
      for (size_t o = 0; o != 4; ++o) {
        ++index;
        if (!items[index].key) {
          indices[msize] = index;
          ++msize;
          items[index] = {key, value};
          return {};
        }
      }
      [[unlikely]] expand();
    }
  }

  void clear() {
    for (size_t i = 0; i != msize; ++i) {
      items[indices[i]].key = {};
    }
    msize = 0;
  }

  Value get(Key key) {
    size_t index = ObjHash()(key) % (allocated - 1);
    for (size_t o = 0; o != 4; ++o) {
      if (items[index + o].key == key) {
        return items[index + o].value;
      }
    }
    throw std::runtime_error("Moodist.deserialize error: object not found");
  }
};

Vector<std::unique_ptr<ObjMap<PyObject*, size_t>>> offsetsFreelist;
Vector<std::unique_ptr<ObjMap<size_t, PyObject*>>> reverseOffsetsFreelist;

struct SerializeObject : SerializeExpandable {
  using Offsets = ObjMap<PyObject*, size_t>;
  std::unique_ptr<Offsets> offsets;

  Vector<py::object> refs;

  SerializeObject() {
    if (offsetsFreelist.empty()) {
      offsetsFreelist.push_back(std::make_unique<Offsets>());
    }
    offsets = offsetsFreelist.pop_front_value();
  }

  ~SerializeObject() {
    offsets->clear();
    offsetsFreelist.push_back(std::move(offsets));
  }

  std::optional<size_t> add(PyObject* obj, size_t offset) {
    auto r = offsets->add(obj, offset);
    if (!r) {
      refs.push_back(py::reinterpret_borrow<py::object>(obj));
    }
    return r;
  }
};

struct DeserializeObject : Deserialize<true> {
  using Offsets = ObjMap<size_t, PyObject*>;
  std::unique_ptr<Offsets> offsets;
  Vector<py::object> refs;

  DeserializeObject(std::string_view view) : Deserialize(view) {
    if (reverseOffsetsFreelist.empty()) {
      reverseOffsetsFreelist.push_back(std::make_unique<Offsets>());
    }
    offsets = reverseOffsetsFreelist.pop_front_value();
  }

  ~DeserializeObject() {
    offsets->clear();
    reverseOffsetsFreelist.push_back(std::move(offsets));
  }

  void store(PyObject* obj, size_t offset) {
    offsets->add(offset, obj);
    refs.push_back(py::reinterpret_borrow<py::object>(obj));
  }

  py::object get(size_t offset) {
    return py::reinterpret_borrow<py::object>(offsets->get(offset));
  }
};

template<typename X, bool checked = true>
[[gnu::always_inline]] [[gnu::hot]] static inline void putInt(X& x, long value) {
  static_assert(sizeof(long) == 8);
  static_assert(sizeof(long) == sizeof(int64_t));
  static_assert(std::is_same_v<long, int64_t>);
  static_assert(std::endian::native == std::endian::little);

  if (checked) {
    x.ensure(16);
  }

  long v32 = (int32_t)value;
  long v16 = (int16_t)value;
  long v8 = (int8_t)value;

  bool i32 = v32 == value;
  bool i16 = v16 == value;
  bool i8 = v8 == value;

#if __x86_64__
  long n2 = 2;
  long n1 = 1;
  long n0 = 0;
  long index;
  asm("mov $3, %0\n"
      "cmp %1, %2\n"
      "cmove %5, %0\n"
      "cmp %1, %3\n"
      "cmove %6, %0\n"
      "cmp %1, %4\n"
      "cmove %7, %0\n"
      : "=&r"(index)
      : "r"(value), "r"(v32), "r"(v16), "r"(v8), "r"(n2), "r"(n1), "r"(n0));
#else
  uint8_t index = 3 - i32 - i16 - i8;
#endif

  uint8_t bytes = 1 << index;

  uint8_t* ptr = (uint8_t*)x.data();
  ptr[0] = pyTypes::int8 + index;
  std::memcpy(ptr + 1, &value, 8);

  x.advance(1 + bytes);
}

template<typename X>
long getint(X& x) {
  pyTypes type;
  x(type);
  if (type == pyTypes::none) {
    return 0;
  }
  if (type == int8) {
    return x.template read<int8_t>();
  }
  if (type == int16) {
    return x.template read<int16_t>();
  }
  asm volatile("");
  if (type == int32) {
    return x.template read<int32_t>();
  }
  if (type == int64) {
    return x.template read<int64_t>();
  }
  throw std::runtime_error(fmt::sprintf("Moodist.serialize: expected int, got type %d", (int)type));
}

} // namespace

template<typename X>
void serialize(X& x, const py::bool_& v) {
  if (v.ptr() == Py_True) {
    x(true);
  } else if (v.ptr() == Py_False) {
    x(false);
  } else {
    throw SerializationError("bad bool\n");
  }
}

template<typename X>
void serialize(X& x, const py::float_& v) {
  x((float)v);
}

template<typename X>
void serialize(X& x, const py::dict& v) {
  putInt(x, v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}
template<typename X>
void serialize(X& x, py::dict& v) {
  size_t n = getint(x);
  for (size_t i = 0; i != n; ++i) {
    auto key = x.template read<py::object>();
    v[key] = x.template read<py::object>();
  }
}

template<typename X>
void serialize(X& x, const py::set& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
void serialize(X& x, py::set& v) {
  size_t n = x.template read<size_t>();
  for (size_t i = 0; i != n; ++i) {
    v.add(x.template read<py::object>());
  }
}

template<typename X>
void serialize(X& x, const py::frozenset& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
void serialize(X& x, py::frozenset& v) {
  py::set s;
  size_t n = x.template read<size_t>();
  for (size_t i = 0; i != n; ++i) {
    s.add(x.template read<py::object>());
  }
  v = std::move(s);
}

template<typename X>
[[gnu::noinline]] void serialize(X& x, const py::list& v) {
  x((size_t)v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
[[gnu::noinline]] void serialize(X& x, py::list& v) {
  size_t n = x.template read<size_t>();
  v = py::list(n);
  for (size_t i = 0; i != n; ++i) {
    v[i] = x.template read<py::object>();
  }
}

template<typename X>
void serialize(X& x, const py::bytes& v) {
  x((std::string_view)v);
}
template<typename X>
void serialize(X& x, py::bytes& v) {
  auto view = x.template read<std::string_view>();
  v = py::bytes(view.data(), view.size());
}

template<typename X>
void serialize(X& x, const py::bytearray& v) {
  x(std::string_view(PyByteArray_AS_STRING(v.ptr()), PyByteArray_GET_SIZE(v.ptr())));
}
template<typename X>
void serialize(X& x, py::bytearray& v) {
  auto view = x.template read<std::string_view>();
  v = py::bytearray(view.data(), view.size());
}

template<typename X>
void serialize(X& x, const py::args& v) {
  x(static_cast<const py::tuple&>(v));
}
template<typename X>
void serialize(X& x, py::args& v) {
  x(static_cast<py::tuple&>(v));
}

template<typename X>
void serialize(X& x, const py::kwargs& v) {
  x(static_cast<const py::dict&>(v));
}
template<typename X>
void serialize(X& x, py::kwargs& v) {
  x(static_cast<py::dict&>(v));
}

struct ConstructorList {
  static constexpr pyTypes type = pyTypes::list;
  static PyObject* make(size_t n) {
    return PyList_New(n);
  }
  static size_t size(PyObject* obj) {
    return PyList_GET_SIZE(obj);
  }
  static PyObject* get(PyObject* obj, size_t index) {
    return PyList_GET_ITEM(obj, index);
  }
  static void set(PyObject* obj, size_t index, PyObject* value) {
    PyList_SET_ITEM(obj, index, value);
  }
};

struct ConstructorVector {
  static constexpr pyTypes type = pyTypes::list;
  static size_t size(const Vector<py::object>& obj) {
    return obj.size();
  }
  static PyObject* get(Vector<py::object>& obj, size_t index) {
    return obj[index].ptr();
  }
};

struct ConstructorTuple {
  static constexpr pyTypes type = pyTypes::tuple;
  static PyObject* make(size_t n) {
    return PyTuple_New(n);
  }
  static size_t size(PyObject* obj) {
    return PyTuple_GET_SIZE(obj);
  }
  static PyObject* get(PyObject* obj, size_t index) {
    return PyTuple_GET_ITEM(obj, index);
  }
  static void set(PyObject* obj, size_t index, PyObject* value) {
    PyTuple_SET_ITEM(obj, index, value);
  }
};

template<typename X, typename Constructor = ConstructorList, typename O = PyObject*>
[[gnu::noinline]] [[gnu::hot]] static inline void putList(X& x, O&& obj) {
  size_t n = Constructor::size(obj);
  x(Constructor::type);
  putInt(x, n);
  size_t i = 0;
#if PY_VERSION_HEX >= 0x030c0000
  constexpr size_t N = 8;
  while (n - i >= N) {
    bool escape = false;
    x.ensure(16 * N);
    for (size_t o = 0; o != N; ++o) {
      PyObject* obj2 = Constructor::get(obj, i + o);
      if (Py_TYPE(obj2) == &PyLong_Type) {
        if (_PyLong_IsCompact((PyLongObject*)obj2)) [[likely]] {
          putInt<X, false>(x, _PyLong_CompactValue((PyLongObject*)obj2));
          continue;
        }
      }
      escape = true;
      i += o;
      break;
    }
    if (escape) {
      break;
    }
    i += N;
  }
#endif
  for (; i != n; ++i) {
    x((py::handle)Constructor::get(obj, i));
  }
}

template<typename X>
static inline void putTuple(X& x, PyObject* obj) {
  putList<X, ConstructorTuple>(x, obj);
}

[[gnu::always_inline]] static inline py::object stealcheck(PyObject* ptr) {
  if (!ptr) [[unlikely]] {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(ptr);
}

[[gnu::always_inline]] static inline py::object stealn(PyObject* ptr) {
  return py::reinterpret_steal<py::object>(ptr);
}

static inline py::object getattr(PyObject* obj, PyObject* attr) {
  PyObject* ptr = PyObject_GetAttr(obj, attr);
  if (!ptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(ptr);
}

template<typename X>
[[gnu::noinline]] static inline void putGlobal(X& x, PyObject* obj) {
  py::object name = getattr(obj, strQualName);
  py::object module = getattr(obj, strModule);

  // log.info("put global name is %s\n", (std::string)(py::str)name);
  // log.info("put global module is %s\n", (std::string)(py::str)module);

  x(pyTypes::global, name, module);
}

struct CacheHash {
  size_t operator()(std::pair<std::string_view, std::string_view> v) {
    return std::hash<std::string_view>()(v.first) ^ std::hash<std::string_view>()(v.second);
  }
};

// std::string_view((char*)PyUnicode_DATA(obj), PyUnicode_GET_LENGTH(obj))
HashMap<std::pair<std::string_view, std::string_view>, PyObject*, CacheHash> cachedGlobals;

template<typename X>
[[gnu::noinline]] static inline py::object getGlobal(X& x) {
  py::object name;
  py::object moduleName;

  x(name, moduleName);

  auto key = std::make_pair(
      std::string_view((char*)PyUnicode_DATA(moduleName.ptr()), PyUnicode_GET_LENGTH(moduleName.ptr())),
      std::string_view((char*)PyUnicode_DATA(name.ptr()), PyUnicode_GET_LENGTH(name.ptr())));
  auto it = cachedGlobals.find(key);
  if (it != cachedGlobals.end()) [[likely]] {
    return py::reinterpret_borrow<py::object>(it->second);
  }

  // log.info("get global name is %s\n", (std::string)(py::str)name);
  // log.info("get global module is %s\n", (std::string)(py::str)moduleName);

  py::object o = stealcheck(PyImport_Import(moduleName.ptr()));
  py::object list = stealcheck(PyUnicode_Split(name.ptr(), strDot, -1));

  size_t n = ConstructorList::size(list.ptr());
  for (size_t i = 0; i != n; ++i) {
    o = getattr(o, ConstructorList::get(list.ptr(), i));
  }

  name.inc_ref();
  moduleName.inc_ref();
  o.inc_ref();

  cachedGlobals[key] = o.ptr();

  return o;
}

[[gnu::always_inline]] [[gnu::hot]] static inline py::object toint(long value) {
  if (value >= intlistRangeBegin && value < intlistRangeEnd) [[likely]] {
    PyObject* o = intlist[value + intlistRange / 2];
    [[assume(o != nullptr)]];
    return py::reinterpret_borrow<py::object>(o);
  }
  return stealcheck(PyLong_FromLong(value));
  // return py::int_(value);
}

[[gnu::always_inline]] [[gnu::hot]] static inline py::object tofloat(double value) {
  return stealcheck(PyFloat_FromDouble(value));
}

template<typename X>
[[gnu::noinline]] static inline void putClassObject(X& x, PyObject* obj) {
  py::object reduce;
  py::object reduceExFunc = getattr(obj, strReduceEx);
  if (!reduceExFunc) {
    py::object reduceFunc = getattr(obj, strReduce);
    reduce = stealcheck(PyObject_CallNoArgs(reduceFunc.ptr()));
  } else {
    reduce = stealcheck(PyObject_CallOneArg(reduceExFunc.ptr(), toint(5).ptr()));
  }
  auto* type = Py_TYPE(reduce.ptr());
  if (type == &PyUnicode_Type) {
    py::object module = getattr(obj, strModule);
    x(pyTypes::global, reduce, module);
    return;
  }
  if (type != &PyTuple_Type) {
    throw std::runtime_error(fmt::sprintf(
        "Moodist.serialize: %s.__reduce__ must return a tuple, but got %s\n",
        (std::string)py::str((PyObject*)Py_TYPE(obj)), (std::string)py::str(reduce.get_type())));
  }
  size_t n = ConstructorTuple::size(reduce.ptr());
  PyObject* func = n < 1 ? nullptr : ConstructorTuple::get(reduce.ptr(), 0);
  PyObject* args = n < 2 ? nullptr : ConstructorTuple::get(reduce.ptr(), 1);
  PyObject* state = n < 3 ? Py_None : ConstructorTuple::get(reduce.ptr(), 2);
  PyObject* listIterator = n < 4 ? Py_None : ConstructorTuple::get(reduce.ptr(), 3);
  PyObject* dictIterator = n < 5 ? Py_None : ConstructorTuple::get(reduce.ptr(), 4);
  PyObject* setstate = n < 6 ? Py_None : ConstructorTuple::get(reduce.ptr(), 5);
  if (!func || !args) {
    throw std::runtime_error(fmt::sprintf(
        "Moodist.serialize: %s.__reduce__ must return a tuple of minimum length 2 (got %d)\n",
        (std::string)py::str((PyObject*)Py_TYPE(obj)), n));
  }

  // log.info("func is %s\n", (std::string)(py::str)((py::handle)func).attr("__name__"));

  x(pyTypes::class_, (py::handle)func, (py::handle)args);

  uint8_t bits = 0;
  if (state != Py_None) {
    bits |= 1;
  }
  if (listIterator != Py_None) {
    bits |= 2;
  }
  if (dictIterator != Py_None) {
    bits |= 4;
  }
  if (setstate != Py_None) {
    bits |= 8;
  }
  x(bits);
  if (bits == 0) {
    return;
  }

  if (bits & 1) {
    x((py::handle)state);
    if (bits & 8) {
      x((py::handle)setstate);
    }
  }

  if (bits & 2) {
    Vector<py::object> l;
    while (true) {
      auto o = stealn(PyIter_Next(listIterator));
      if (!o) {
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        break;
      }
      l.push_back(o);
    }
    putList<X, ConstructorVector>(x, l);
  }

  if (bits & 4) {
    auto sizeOffset = x.tell();
    x((size_t)0);
    size_t n = 0;
    while (true) {
      auto o = stealn(PyIter_Next(dictIterator));
      if (!o) {
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        break;
      }
      if (Py_TYPE(o.ptr()) != &PyTuple_Type || ConstructorTuple::size(o.ptr()) != 2) {
        throw std::runtime_error("Moodist serialize: dict item is not a tuple of size 2");
      }
      x(ConstructorTuple::get(o.ptr(), 0), ConstructorTuple::get(o.ptr(), 1));
      ++n;
    }
    std::memcpy((char*)x.data() + x.tell() - x.tell() + sizeOffset, &n, sizeof(n));
  }
}

template<typename X>
[[gnu::noinline]] static inline py::object getClassObject(X& x, size_t offset, bool store) {
  py::object func;
  py::object args;
  x(func, args);
  py::object r = py::reinterpret_steal<py::object>(PyObject_Call(func.ptr(), args.ptr(), nullptr));
  if (!r) {
    throw py::error_already_set();
  }
  if (store) {
    ((DeserializeObject&)x).store(r.ptr(), offset);
  }
  uint8_t bits;
  x(bits);
  if (bits & 1) {
    py::object state;
    x(state);
    CHECK(state.ptr() != Py_None);
    if (bits & 8) {
      py::object setstate;
      x(setstate);
      stealcheck(PyObject_CallOneArg(setstate.ptr(), state.ptr()));
    } else {
      PyObject* stateobj = state.ptr();
      PyObject* slotobj = Py_None;
      if (Py_TYPE(stateobj) == &PyTuple_Type) {
        slotobj = ConstructorList::get(state.ptr(), 1);
        stateobj = ConstructorList::get(state.ptr(), 0);
      }
      if (stateobj != Py_None) {
        py::object dict = getattr(r, strDict);

        Py_ssize_t i = 0;
        PyObject* key;
        PyObject* value;
        while (PyDict_Next(stateobj, &i, &key, &value)) {
          if (PyObject_SetItem(dict.ptr(), key, value)) {
            throw py::error_already_set();
          }
        }
      }
      if (slotobj != Py_None) {
        Py_ssize_t i = 0;
        PyObject* key;
        PyObject* value;
        while (PyDict_Next(slotobj, &i, &key, &value)) {
          if (PyObject_SetAttr(r.ptr(), key, value)) {
            throw py::error_already_set();
          }
        }
      }
    }
  }
  if (bits & 2) {
    py::object extendlist;
    x(extendlist);
    CHECK(extendlist.ptr() != Py_None);
    stealcheck(PyObject_CallOneArg(getattr(r.ptr(), strExtend).ptr(), extendlist.ptr()));
  }

  if (bits & 4) {
    size_t nitems = x.template read<size_t>();
    CHECK(nitems);
    while (nitems) {
      py::object key;
      py::object value;
      x(key, value);
      if (PyObject_SetItem(r.ptr(), key.ptr(), value.ptr())) {
        throw py::error_already_set();
      }
      --nitems;
    }
  }

  return r;
}

template<typename X>
[[gnu::noinline]] [[gnu::hot]] static inline void serialize2(X& x, PyObject* obj, PyTypeObject* type) {

  auto it = ((SerializeObject&)x).add(obj, x.tell());
  if (it) {
    // log.info("reuse object %p at offset %d (stored at offset %d)\n", (void*)obj, it.first->second, x.tell());
    *((uint8_t*)x.data() - x.tell() + *it) |= 0x80;
    x(pyTypes::offset, *it);
    return;
  }
  // log.info("new object %p at offset %d\n", (void*)obj, x.tell());
  // CHECK(x.tell() == *it);

  if (type == &PyList_Type) {
    putList(x, obj);
    return;
  }
  if (type == &PyTuple_Type) {
    putTuple(x, obj);
    return;
  }

  if (type == &PyDict_Type) {
    x(pyTypes::dict, py::reinterpret_borrow<py::dict>(obj));
    return;
  }

  if (type == &PyUnicode_Type) {
    x(pyTypes::unicode, (uint8_t)PyUnicode_KIND(obj));
    // PyUnicode_GET_LENGTH returns code points, multiply by kind to get bytes
    x(std::string_view((char*)PyUnicode_DATA(obj), PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj)));
    return;
  }

  if (type == &PyBytes_Type) {
    x(pyTypes::bytes, py::reinterpret_borrow<py::bytes>(obj));
    return;
  }

  if (type == &PyByteArray_Type) {
    x(pyTypes::bytearray, py::reinterpret_borrow<py::bytearray>(obj));
    return;
  }

  if (type == &PySet_Type) {
    x(pyTypes::set, py::reinterpret_borrow<py::set>(obj));
    return;
  }
  if (type == &PyFrozenSet_Type) {
    x(pyTypes::frozenset, py::reinterpret_borrow<py::frozenset>(obj));
    return;
  }

  if (type == &PyFunction_Type || type == &PyType_Type) {
    putGlobal(x, obj);
    return;
  }

  putClassObject(x, obj);
}

template<typename X>
[[gnu::always_inline]] [[gnu::hot]] static inline void serialize(X& x, const py::handle& v) {
  PyObject* obj = v.ptr();

  auto* type = Py_TYPE(obj);

  if (obj == Py_True) {
    x(pyTypes::bool_true);
    return;
  }
  if (obj == Py_False) {
    x(pyTypes::bool_false);
    return;
  }
  if (obj == Py_None) {
    x(pyTypes::none);
    return;
  }

  if (type == &PyLong_Type) {
#if PY_VERSION_HEX >= 0x030c0000
    if (_PyLong_IsCompact((PyLongObject*)obj)) [[likely]] {
      long value = _PyLong_CompactValue((PyLongObject*)obj);
#else
    int overflow = 0;
    long value = PyLong_AsLongAndOverflow(obj, &overflow);
    if (!overflow) [[likely]] {
#endif
      putInt(x, value);
    } else {
      size_t nbits = _PyLong_NumBits(obj);
      size_t nbytes = (nbits + 7) / 8 + 1;
      IVector<char> tmp(nbytes);
#if PY_VERSION_HEX >= 0x030d0000
      if (_PyLong_AsByteArray((PyLongObject*)obj, (unsigned char*)tmp.data(), nbytes, 1, 1, 1)) {
#else
      if (_PyLong_AsByteArray((PyLongObject*)obj, (unsigned char*)tmp.data(), nbytes, 1, 1)) {
#endif
        throw std::runtime_error("_PyLong_AsByteArray failed");
      }
      x(pyTypes::bigint, std::string_view(tmp.data(), tmp.size()));
    }
    return;
  }
  if (type == &PyFloat_Type) {
    x(pyTypes::float_, PyFloat_AS_DOUBLE(obj));
    return;
  }

  serialize2(x, obj, type);
}

template<typename X, typename Constructor = ConstructorList>
[[gnu::noinline]] [[gnu::hot]] static inline py::object getList(X& x, size_t offset, bool store) {
  size_t n = getint(x);
  py::object r = py::reinterpret_steal<py::object>(Constructor::make(n));
  if (store) {
    ((DeserializeObject&)x).store(r.ptr(), offset);
  }
  size_t i = 0;
  constexpr size_t N = 8;
  while (n - i >= N && x.remaining() >= 16 * N) {
    bool escape = false;
    for (size_t j = 0; j != N;) {
      pyTypes type;
      std::memcpy(&type, x.data(), sizeof(type));
      if (type <= float_) {
        x.consumeNoCheck(1);
        if (type <= bool_false) {
          PyObject* arr[3] = {Py_None, Py_True, Py_False};
          PyObject* o = arr[type];
          Py_INCREF(o);
          Constructor::set(r.ptr(), i, o);
          ++i;
          ++j;
          continue;
        }
        int64_t value;
        static_assert(sizeof(int64_t) == 8);
        std::memcpy(&value, x.data(), 8);
        // int8_t v8 = value;
        // int16_t v16 = value;
        // int32_t v32 = value;
        if (type == float_) {
          double fv;
          static_assert(sizeof(double) == 8);
          std::memcpy(&fv, &value, sizeof(double));
          x.consumeNoCheck(8);
          Constructor::set(r.ptr(), i, tofloat(fv).release().ptr());
          ++i;
          ++j;
          while (j != N) {
            pyTypes ntype;
            std::memcpy(&ntype, x.data(), sizeof(ntype));
            if (ntype != type) {
              break;
            }
            std::memcpy(&fv, (char*)x.data() + 1, 8);
            x.consumeNoCheck(1 + 8);
            Constructor::set(r.ptr(), i, tofloat(fv).release().ptr());
            ++i;
            ++j;
          }
          continue;
        }
        uint8_t nbytesarr[4] = {1, 2, 4, 8};
        uint8_t shiftarr[4] = {56, 48, 32, 0};
        long index = type - int8;
        uint8_t nbytes = nbytesarr[index];
        uint8_t shift = shiftarr[index];
        x.consumeNoCheck(nbytes);
        value = value << shift >> shift;
        Constructor::set(r.ptr(), i, toint(value).release().ptr());
        // int64_t arr[4] = {v8, v16, v32, value};
        // PyList_SET_ITEM(r.ptr(), i, toint(arr[type - int8]).release().ptr());
        ++i;
        ++j;
        while (j != N) {
          pyTypes ntype;
          std::memcpy(&ntype, x.data(), sizeof(ntype));
          if (ntype != type) {
            break;
          }
          std::memcpy(&value, (char*)x.data() + 1, 8);
          x.consumeNoCheck(1 + nbytes);
          value = value << shift >> shift;
          Constructor::set(r.ptr(), i, toint(value).release().ptr());
          ++i;
          ++j;
        }
        continue;
      }
      escape = true;
      break;
    }
    if (escape) {
      break;
    }
  }
  for (; i != n; ++i) {
    Constructor::set(r.ptr(), i, deserialize(x).release().ptr());
  }
  return r;
}

template<typename X>
static inline py::object getTuple(X& x, size_t offset, bool store) {
  return getList<X, ConstructorTuple>(x, offset, store);
}

template<typename X>
[[gnu::noinline]] static inline py::dict getDict(X& x, size_t offset, bool store) {
  size_t n = getint(x);
  py::dict r;
  if (store) {
    ((DeserializeObject&)x).store(r.ptr(), offset);
  }
  for (size_t i = 0; i != n; ++i) {
    auto key = deserialize(x);
    auto value = deserialize(x);
    if (PyDict_SetItem(r.ptr(), key.ptr(), value.ptr())) {
      throw py::error_already_set();
    }
  }
  return r;
}

template<typename X>
[[gnu::always_inline]] [[gnu::hot]] static inline py::object deserialize2(X& x, pyTypes type) {
  bool store = false;
  size_t offset = x.tell() - 1 + 1;
  if (type & 0x80) {
    store = true;
    type = (pyTypes)(type & 0x7f);
  }
  py::object r;
  switch (type) {
  case pyTypes::dict:
    return getDict(x, offset, store);
    break;
  case pyTypes::str:
    r = x.template read<py::str>();
    break;
  case pyTypes::bigint: {
    auto view = x.template read<std::string_view>();
    auto* obj = _PyLong_FromByteArray((unsigned char*)view.data(), view.size(), 1, 1);
    if (!obj) {
      throw py::error_already_set();
    }
    r = py::reinterpret_steal<py::object>(obj);
    break;
  }
  case pyTypes::list:
    return getList(x, offset, store);
  case pyTypes::tuple:
    return getTuple(x, offset, store);
  case pyTypes::unicode: {
    uint8_t kind;
    std::string_view view;
    x(kind, view);
    // view.size() is in bytes, PyUnicode_FromKindAndData expects code point count
    auto* obj = PyUnicode_FromKindAndData(kind, view.data(), view.size() / kind);
    if (!obj) {
      throw py::error_already_set();
    }
    r = py::reinterpret_steal<py::object>(obj);
    break;
  }
  case pyTypes::global:
    r = getGlobal(x);
    break;
  case pyTypes::class_:
    return getClassObject(x, offset, store);
  case pyTypes::offset:
    r = ((DeserializeObject&)x).get(std::max(1 + x.template read<size_t>(), (size_t)1));
    break;
  case pyTypes::bytes:
    r = x.template read<py::bytes>();
    break;
  case pyTypes::bytearray:
    r = x.template read<py::bytearray>();
    break;
  case pyTypes::set:
    r = x.template read<py::set>();
    break;
  case pyTypes::frozenset:
    r = py::frozenset(x.template read<py::set>());
    break;
  default:
    throw SerializationError("Can't deserialize python type (unknown type " + std::to_string(type) + ")");
  }
  if (store) {
    ((DeserializeObject&)x).store(r.ptr(), offset);
  }
  return r;
}

template<typename X>
[[gnu::always_inline]] [[gnu::hot]] static inline py::object deserialize(X& x) {
  using enum pyTypes;
  pyTypes type;
  x(type);
  if (type <= float_) {
    if (type <= bool_false) {
      PyObject* arr[3] = {Py_None, Py_True, Py_False};
      return py::reinterpret_borrow<py::object>(arr[type]);
    }
    if (type == int8) {
      return toint(x.template read<int8_t>());
    }
    if (type == int16) {
      return toint(x.template read<int16_t>());
    }
    asm volatile("");
    if (type == int32) {
      return toint(x.template read<int32_t>());
    }
    if (type == int64) {
      return toint(x.template read<int64_t>());
    }
    if (type == float_) {
      return tofloat(x.template read<double>());
    }
    CHECK(false);
  }
  return deserialize2(x, type);
}

template<typename X>
static inline void serialize(X& x, py::object& v) {
  v = deserialize(x);
}

template<typename X>
void serialize(X& x, const py::tuple& v) {
  size_t n = v.size();
  x(n);
  for (auto& v2 : v) {
    x(v2);
  }
}
template<typename X>
void serialize(X& x, py::tuple& v) {
  size_t n = x.template read<size_t>();
  v = py::tuple(n);
  for (size_t i = 0; i != n; ++i) {
    v[i] = x.template read<py::object>();
  }
}

template<typename... T>
[[gnu::always_inline]] [[gnu::warn_unused_result]] inline BufferHandle serializeObjectToBuffer(const T&... v) {
  SerializeObject x;
  x.buffer = makeBuffer(0x80 - sizeof(Buffer));
  (x(v), ...);
  x.buffer->msize = x.size;
  return std::move(x.buffer);
}

template<typename... T>
[[gnu::always_inline]] inline void deserializeObjectFromBuffer(const void* ptr, size_t len, T&... result) {
  DeserializeObject x(std::string_view{(const char*)ptr, len});
  x(result...);
  if (x.des.buf().size() != 0) {
    throw SerializationError("deserializeObject: " + std::to_string(x.des.buf().size()) + " trailing bytes");
  }
}

torch::Tensor serializeObject(py::object o) {
  std::call_once(globalsInitFlag, globalsInit);

  auto buffer = serializeObjectToBuffer(o);
  void* data = buffer->data();
  size_t size = buffer->size();
  SharedBufferHandle sharedBuffer(buffer.release());
  return torch::from_blob(
      data, {(int64_t)size}, [sharedBuffer = std::move(sharedBuffer)](void*) {},
      torch::TensorOptions().dtype(torch::kUInt8));
}

py::object deserializeObject(torch::Tensor t) {
  std::call_once(globalsInitFlag, globalsInit);

  py::object o;
  deserializeObjectFromBuffer(t.data_ptr(), t.itemsize() * t.numel(), o);
  return o;
}

} // namespace moodist
