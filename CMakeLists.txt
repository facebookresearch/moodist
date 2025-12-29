cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(moodist LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_ALLOCATOR_GUARDS "Enable allocator guard bytes" OFF)
set(COMPILE_FLAGS "-Wfatal-errors -Wno-unused-result -Bsymbolic -march=x86-64-v3 -mtls-dialect=gnu2 -U_FORTIFY_SOURCE -fno-stack-protector")
if(ENABLE_ASAN)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0 ${COMPILE_FLAGS}")
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} ${COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -gz")
set(CMAKE_C_FLAGS_RELWITHDEBINFO
    "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -gz")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_SKIP_RPATH TRUE)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND python -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
message(STATUS "PyTorch is at ${TorchPath}")

# We're forced to inherit from a class defined in pytorch, whose pybind code is also in pytorch.
# pybind then requires the abi of our class and the parent class to be the same.
# (pybind enforces this by a string lookup, where the string is composed of compiler, abi version etc)
set(ABI_VERSION "auto" CACHE STRING "The ABI version that pytorch was built with")
set(USE_CXX1_ABI "auto" CACHE STRING "_GLIBCXX_USE_CXX11_ABI option that pytorch was built with")

if (ABI_VERSION STREQUAL "auto" OR USE_CXX1_ABI STREQUAL "auto")
  message(STATUS "Auto-detecting pytorch abi")
  foreach(abi IN ITEMS 19 18 16 11 13 20 17 15 14 12)
    foreach (cxx IN ITEMS 1 0)
      set(dir ${CMAKE_CURRENT_BINARY_DIR}/detect_abi_${abi}_${cxx})
      try_compile(success PROJECT detect_abi SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/detect_abi BINARY_DIR ${dir} CMAKE_FLAGS -DABI_VERSION=${abi} -DUSE_CXX1_ABI=${cxx} -DTorchPath=${TorchPath} OUTPUT_VARIABLE stdout)
      #message(STATUS "${stdout}")
      if (success)
        execute_process(
          COMMAND python -c "import torch, moodist_detect_abi"
          WORKING_DIRECTORY ${dir}
          RESULT_VARIABLE exitcode
        )
        #message(STATUS "exitcode for ${dir} is ${exitcode}")
        if (exitcode EQUAL 0)
          set(ABI_VERSION ${abi} CACHE STRING "" FORCE)
          set(USE_CXX1_ABI ${cxx} CACHE STRING "" FORCE)
          break()
        endif()
      endif()
    endforeach()
    if (NOT (ABI_VERSION STREQUAL "auto" OR USE_CXX1_ABI STREQUAL "auto"))
      break()
    endif()
  endforeach()

  if (ABI_VERSION STREQUAL "auto" OR USE_CXX1_ABI STREQUAL "auto")
    message(FATAL_ERROR "Failed to auto-detect abi version for pytorch. Set ABI_VERSION and USE_CXX1_ABI to the appropriate values.")
  endif()
endif()

message(STATUS "Using -fabi-version=${ABI_VERSION}")
message(STATUS "Using -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI}")

# Note: -fabi-version and _GLIBCXX_USE_CXX11_ABI are set per-target, not globally
# Core library uses system default ABI, wrapper matches PyTorch's ABI

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_INCLUDE_DIRS}, lib: ${CUDAToolkit_LIBRARY_DIR}")

find_library(CUDADEVRT libcudadevrt.a HINTS ${CUDAToolkit_LIBRARY_DIR})

find_package(Threads REQUIRED)

add_subdirectory(pybind11)

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

set(ENABLE_STATIC ON CACHE BOOL "")
set(ENABLE_RESOLVE_NEIGH OFF CACHE BOOL "")
set(NO_PYVERBS ON CACHE BOOL "")
set(NO_MAN_PAGES ON CACHE BOOL "")
set(NO_COMPAT_SYMS ON CACHE BOOL "")
set(HAVE_FUNC_ATTRIBUTE_SYMVER "0" CACHE STRING "" FORCE)
add_subdirectory(rdma-core EXCLUDE_FROM_ALL)

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

# Core library sources - uses wrapper interfaces, no direct PyTorch
set(core_sources
  core/allgather.cc
  core/allocator.cc
  core/alltoall.cc
  core/async.cc
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
  core/queue.cc
  core/rdma_ib.cc
  core/rdma_tcp.cc
  core/reduce_scatter.cc
  core/reduce_scatter_direct.cc
  core/serialize_object.cc
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
    CUDA::cuda_driver
    CUDA::nvml
    CUDA::nvrtc
    Threads::Threads
    anl
    kern-abi  # rdma-core kernel ABI
  )
  target_include_directories(moodist SYSTEM PRIVATE
    fmt/include
    rdma-core
    ${CMAKE_CURRENT_BINARY_DIR}/rdma-core/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}  # For serialize_object.cc
    ${Python_INCLUDE_DIRS}  # For serialize_object.cc
  )
  target_include_directories(moodist PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/api
  )
  target_compile_definitions(moodist PRIVATE
    CUDADEVRT_PATH="${CUDADEVRT}"
    _STATIC_LIBRARY_BUILD_
    $<$<BOOL:${ENABLE_ALLOCATOR_GUARDS}>:MOODIST_ALLOCATOR_GUARDS>
  )
  set_property(TARGET moodist PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Wrapper library - compiled per PyTorch version, uses dlopen for core
set(wrapper_sources
  wrapper/pybind.cc
  wrapper/store_wrapper.cc
  wrapper/serialize_wrapper.cc
  wrapper/moodist_loader.cc
  wrapper/allocator_wrapper.cc
  wrapper/cpu_allocator_wrapper.cc
  wrapper/processgroup_wrapper.cc
  wrapper/queue_wrapper.cc
)

pybind11_add_module(_C ${wrapper_sources})
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
  ${CMAKE_CURRENT_SOURCE_DIR}/api
)
target_compile_options(_C PRIVATE -fabi-version=${ABI_VERSION})
target_compile_definitions(_C PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI}
  MOODIST_WRAPPER  # Enable direct wrapper function calls in _C.so
)

set_property(TARGET _C PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

# Ensure moodist is built when building _C (even though we don't link directly)
if(NOT MOODIST_PREBUILT_CORE)
  add_dependencies(_C moodist)
endif()
