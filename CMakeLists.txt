cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(moodist LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_ALLOCATOR_GUARDS "Enable allocator guard bytes" OFF)
option(ENABLE_STDLIB_ALLOCATOR "Use stdlib malloc instead of custom allocator" OFF)

# Build component options - allows building subsets of the project
option(MOODIST_BUILD_CORE "Build libmoodist.so core library (requires CUDA)" ON)
option(MOODIST_BUILD_WRAPPER "Build _C.so PyTorch wrapper (requires Python + PyTorch)" ON)
option(MOODIST_BUILD_SERIALIZE "Build _serialize.so library (requires Python)" ON)
set(COMPILE_FLAGS "-Wfatal-errors -Wno-unused-result -Bsymbolic -march=x86-64-v3 -U_FORTIFY_SOURCE -fno-stack-protector")
if(ENABLE_ASAN)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -static-libasan")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -static-libasan")
endif()
if(ENABLE_TSAN)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
endif()
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0 ${COMPILE_FLAGS}")
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} ${COMPILE_FLAGS}")

option(MINIMAL_DEBUG "Use -g1 for smaller binaries (less debug info)" OFF)
if(MINIMAL_DEBUG)
  set(DEBUG_FLAG "-g1")
else()
  set(DEBUG_FLAG "-g")
endif()

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 ${DEBUG_FLAG} -DNDEBUG -gz")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 ${DEBUG_FLAG} -DNDEBUG -gz")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_SKIP_RPATH TRUE)

# Python is required for wrapper and serialize
if(MOODIST_BUILD_WRAPPER OR MOODIST_BUILD_SERIALIZE)
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
endif()

# PyTorch is only required for wrapper
if(MOODIST_BUILD_WRAPPER)
  execute_process(
      COMMAND python -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
      OUTPUT_VARIABLE TorchPath
  )
  message(STATUS "PyTorch is at ${TorchPath}")

  # Detect _GLIBCXX_USE_CXX11_ABI from PyTorch (affects std::string layout)
  set(USE_CXX1_ABI "auto" CACHE STRING "_GLIBCXX_USE_CXX11_ABI option that pytorch was built with")

  if (USE_CXX1_ABI STREQUAL "auto")
    execute_process(
      COMMAND python -c "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0, end='')"
      OUTPUT_VARIABLE USE_CXX1_ABI
      RESULT_VARIABLE cxx_abi_result
    )
    if (NOT cxx_abi_result EQUAL 0)
      message(FATAL_ERROR "Failed to detect _GLIBCXX_USE_CXX11_ABI from PyTorch")
    endif()
    set(USE_CXX1_ABI ${USE_CXX1_ABI} CACHE STRING "" FORCE)
    message(STATUS "Detected _GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI} from PyTorch")
  endif()

  # Note: -fabi-version detection removed - not needed for stable API bindings
  # (ABI version only affects name mangling for templates/lambdas, not layout of types we use)

  message(STATUS "Using -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI}")
endif()

# CUDA is required for core and wrapper (not serialize)
if(MOODIST_BUILD_CORE OR MOODIST_BUILD_WRAPPER)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA: ${CUDAToolkit_INCLUDE_DIRS}, lib: ${CUDAToolkit_LIBRARY_DIR}")

  find_library(CUDADEVRT libcudadevrt.a HINTS ${CUDAToolkit_LIBRARY_DIR})
endif()

find_package(Threads REQUIRED)

# pybind11 is only needed for serialize library
if(MOODIST_BUILD_SERIALIZE)
  add_subdirectory(pybind11)
endif()

# Generate build magic for API version checking
# Random value generated at configure time - pass -DMOODIST_BUILD_MAGIC=0x... to override
set(MOODIST_BUILD_MAGIC "" CACHE STRING "Build magic for API verification (auto-generated if empty)")
if(NOT MOODIST_BUILD_MAGIC)
  string(RANDOM LENGTH 16 ALPHABET "0123456789abcdef" RANDOM_HEX)
  set(MOODIST_BUILD_MAGIC "0x${RANDOM_HEX}ULL" CACHE STRING "" FORCE)
endif()
message(STATUS "Build magic: ${MOODIST_BUILD_MAGIC}")

# Pass build magic to both libraries
add_compile_definitions(MOODIST_BUILD_MAGIC=${MOODIST_BUILD_MAGIC})

# rdma-core is only needed for core library
if(MOODIST_BUILD_CORE AND NOT MOODIST_PREBUILT_CORE)
  set(ENABLE_STATIC ON CACHE BOOL "")
  set(ENABLE_RESOLVE_NEIGH OFF CACHE BOOL "")
  set(NO_PYVERBS ON CACHE BOOL "")
  set(NO_MAN_PAGES ON CACHE BOOL "")
  set(NO_COMPAT_SYMS ON CACHE BOOL "")
  set(HAVE_FUNC_ATTRIBUTE_SYMVER "0" CACHE STRING "" FORCE)
  add_subdirectory(rdma-core EXCLUDE_FROM_ALL)
endif()

set(ibverbs_sources
  cmd.c cmd_ah.c cmd_counters.c cmd_cq.c cmd_device.c cmd_dm.c cmd_fallback.c cmd_flow.c
  cmd_flow_action.c cmd_ioctl.c cmd_mr.c cmd_mw.c cmd_pd.c cmd_qp.c cmd_rwq_ind.c cmd_srq.c
  cmd_wq.c cmd_xrcd.c device.c enum_strs.c ibdev_nl.c init.c marshall.c memory.c
  static_driver.c sysfs.c verbs.c dummy_ops.c neigh.c
)
list(TRANSFORM ibverbs_sources PREPEND rdma-core/libibverbs/)
set(util_sources
  bitmap.c cl_map.c interval_set.c node_name_map.c open_cdev.c rdma_nl.c util.c
)
list(TRANSFORM util_sources PREPEND rdma-core/util/)
set(mlx5_sources
  buf.c cq.c dbrec.c dr_action.c dr_arg.c dr_buddy.c dr_crc32.c dr_dbg.c
  dr_devx.c dr_domain.c dr_icm_pool.c dr_matcher.c dr_ptrn.c dr_rule.c dr_send.c dr_ste.c
  dr_ste_v0.c dr_ste_v1.c dr_ste_v2.c dr_ste_v3.c dr_table.c dr_vports.c mlx5.c mlx5_vfio.c
  qp.c srq.c verbs.c
)
list(TRANSFORM mlx5_sources PREPEND rdma-core/providers/mlx5/)

set(efa_sources
  efa.c verbs.c
)
list(TRANSFORM efa_sources PREPEND rdma-core/providers/efa/)

set(rdma_sources
  ${ibverbs_sources}
  ${util_sources}
  ${mlx5_sources}
  ${efa_sources}
)

# Option to use pre-built core library (for multi-PyTorch version builds)
set(MOODIST_PREBUILT_CORE "" CACHE FILEPATH "Path to pre-built libmoodist.so (skip building core)")

# Core library - only build if requested
if(MOODIST_BUILD_CORE OR MOODIST_BUILD_WRAPPER)

# Core library sources - uses wrapper interfaces, no direct PyTorch
set(core_sources
  core/allgather.cc
  core/allocator.cc
  core/alltoall.cc
  core/async.cc
  core/buffer_api.cc
  core/connection.cc
  core/cpu_allocator.cc
  core/cputhread.cc
  core/group.cc
  core/ib_common.cc
  core/internal_allocator.cc
  core/ipc_mapper.cc
  core/kernels.cc
  core/moodist_api.cc
  core/numa.cc
  core/cuda_loader.cc
  core/queue.cc
  core/rdma_ib.cc
  core/rdma_tcp.cc
  core/reduce_scatter.cc
  core/reduce_scatter_direct.cc
  core/setup_comms.cc
  core/socket.cc
  core/store.cc
  core/tcpdev.cc
  core/processgroup.cc
  # Third-party:
  fmt/src/format.cc
  ${rdma_sources}
)

if(MOODIST_PREBUILT_CORE)
  # Use pre-built core library (for subsequent PyTorch version builds)
  message(STATUS "Using pre-built core library: ${MOODIST_PREBUILT_CORE}")
  add_library(moodist SHARED IMPORTED)
  set_target_properties(moodist PROPERTIES
    IMPORTED_LOCATION ${MOODIST_PREBUILT_CORE}
  )
else()
  # Build core library from source
  add_library(moodist SHARED ${core_sources})
  target_link_libraries(moodist PRIVATE
    Threads::Threads
    kern-abi  # rdma-core kernel ABI
    dl  # For dlsym (CUDA/NVML/NVRTC dynamic loading)
  )
  target_include_directories(moodist SYSTEM PRIVATE
    fmt/include
    rdma-core
    ${CMAKE_CURRENT_BINARY_DIR}/rdma-core/include
  )
  target_include_directories(moodist PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}  # For api/ includes (e.g., #include "api/types.h")
  )
  target_compile_definitions(moodist PRIVATE
    CUDADEVRT_PATH="${CUDADEVRT}"
    _STATIC_LIBRARY_BUILD_
    $<$<BOOL:${ENABLE_ALLOCATOR_GUARDS}>:MOODIST_ALLOCATOR_GUARDS>
    $<$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_TSAN}>,$<BOOL:${ENABLE_STDLIB_ALLOCATOR}>>:MOODIST_USE_STDLIB_ALLOCATOR>
  )
  set_property(TARGET moodist PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

endif() # MOODIST_BUILD_CORE OR MOODIST_BUILD_WRAPPER

# Wrapper library - compiled per PyTorch version, uses dlopen for core
# Using stable API (Py_LIMITED_API) - no pybind11
if(MOODIST_BUILD_WRAPPER)
set(wrapper_sources
  wrapper/module.cc
  wrapper/store_wrapper.cc
  wrapper/moodist_loader.cc
  wrapper/processgroup_wrapper.cc
  wrapper/queue_wrapper.cc
  wrapper/allocator_wrapper.cc
  wrapper/cpu_allocator_wrapper.cc
  wrapper/serialize_wrapper.cc
)

# Use Python's extension module instead of pybind11
# USE_SABI 3.10 enables stable ABI and sets correct .abi3.so suffix
Python_add_library(_C MODULE USE_SABI 3.10 ${wrapper_sources})
target_link_libraries(_C PRIVATE
  ${TorchPath}/lib/libtorch_python.so
  Threads::Threads
  dl  # For dlopen/dlsym
)
target_include_directories(_C SYSTEM PRIVATE
  fmt/include
  ${TorchPath}/include
  ${TorchPath}/include/torch/csrc/api/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_include_directories(_C PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/wrapper
  ${CMAKE_CURRENT_SOURCE_DIR}  # For api/ includes (e.g., #include "api/types.h")
)
target_compile_definitions(_C PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI}
  MOODIST_WRAPPER  # Enable direct wrapper function calls in _C.so
)

set_property(TARGET _C PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

endif() # MOODIST_BUILD_WRAPPER

# Serialize library - compiled per Python version (uses pybind11 internally)
# NOT a Python module - exports C functions that _C loads via dlsym
# Does NOT link against libmoodist.so - uses CoreApi function pointers instead
# Named with Python version suffix (e.g., lib_serialize.cpython-311.so)
if(MOODIST_BUILD_SERIALIZE)
add_library(_serialize SHARED
  wrapper/serialize/serialize_object.cc
)
# Set output name to include Python version (e.g., _serialize.cpython-311)
set_target_properties(_serialize PROPERTIES
  OUTPUT_NAME "_serialize.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}"
)
target_link_libraries(_serialize PRIVATE
  pybind11::module  # For pybind11 headers only (not module init)
)
target_include_directories(_serialize SYSTEM PRIVATE
  fmt/include
  ${Python_INCLUDE_DIRS}
)
target_include_directories(_serialize PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/serialize
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(_serialize PRIVATE
  FMT_HEADER_ONLY
)
set_property(TARGET _serialize PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

endif() # MOODIST_BUILD_SERIALIZE
