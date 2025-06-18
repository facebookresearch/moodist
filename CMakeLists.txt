cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(moodist)

project(moodist LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)
set(COMPILE_FLAGS "-Wfatal-errors -Wno-unused-result -Bsymbolic -march=x86-64-v3 -mtls-dialect=gnu2")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0 ${COMPILE_FLAGS}")
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} ${COMPILE_FLAGS}")

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
  foreach(abi IN ITEMS 16 11 13 19 20 18 17 15 14 12)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=${ABI_VERSION}")

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA includes are ${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDAToolkit_LIBRARY_DIR is ${CUDAToolkit_LIBRARY_DIR}")

find_library(CUDADEVRT libcudadevrt.a HINTS ${CUDAToolkit_LIBRARY_DIR})

message(STATUS "CUDADEVRT is ${CUDADEVRT}")

find_package(Threads REQUIRED)

add_subdirectory(pybind11)

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

pybind11_add_module(_C
  processgroup.cc
  group.cc
  setup_comms.cc
  connection.cc
  socket.cc 
  async.cc
  ipc_mapper.cc
  cputhread.cc
  pybind.cc
  ib_common.cc
  allgather.cc
  kernels.cc
  reduce_scatter.cc
  reduce_scatter_direct.cc
  allocator.cc
  queue.cc
  cpu_allocator.cc
  numa.cc
  alltoall.cc
  internal_allocator.cc
  fmt/src/format.cc
  ${rdma_sources}
)
if (TARGET CUDA::nvrtc_static AND TARGET CUDA::nvrtc_builtins_static AND TARGET CUDA::nvml_static)
  target_link_libraries(_C PRIVATE CUDA::nvrtc_static CUDA::nvrtc_builtins_static CUDA::nvml_static)
else()
  target_link_libraries(_C PRIVATE CUDA::nvrtc CUDA::nvml)
  if (DEFINED IS_BUILDING_WHEEL)
    message(FATAL_ERROR "Could not find static cuda libraries. Refusing to continue as we are building a wheel.")
  endif()
  message(WARNING "Linking dynamically to cuda libraries as the static libraries could not be found.")
  message(WARNING "Moodist may throw runtime errors if there is a cuda version mismatch between build and runtime.")
endif()
target_link_libraries(_C PRIVATE
  CUDA::cuda_driver
  ${TorchPath}/lib/libtorch_python.so
  Threads::Threads
  anl
  kern-abi
)
target_include_directories(_C SYSTEM PRIVATE
  fmt/include
  rdma-core
  ${CMAKE_CURRENT_BINARY_DIR}/rdma-core/include
  ${TorchPath}/include
  ${TorchPath}/include/torch/csrc/api/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(_C PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI})
target_compile_definitions(_C PRIVATE CUDADEVRT_PATH="${CUDADEVRT}")
target_compile_definitions(_C PRIVATE _STATIC_LIBRARY_BUILD_)

set_property(TARGET _C PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
