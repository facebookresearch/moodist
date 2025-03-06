cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(moodist)

project(moodist LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wno-unused-result -ftemplate-backtrace-limit=0 -Bsymbolic -march=x86-64-v3")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
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
set(ABI_VERSION "11" CACHE STRING "The ABI version that pytorch was built with")
if(ABI_VERSION)
  message(STATUS "using -fabi-version=${ABI_VERSION}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=${ABI_VERSION}")
else()
  message(STATUS "not using -fabi-version")
endif()

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA includes are ${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDAToolkit_LIBRARY_DIR is ${CUDAToolkit_LIBRARY_DIR}")

find_library(CUDADEVRT libcudadevrt.a HINTS ${CUDAToolkit_LIBRARY_DIR})

message(STATUS "CUDADEVRT is ${CUDADEVRT}")

find_package(Threads REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

set(ENABLE_STATIC ON CACHE BOOL "")
set(ENABLE_RESOLVE_NEIGH OFF CACHE BOOL "")
set(RDMA_STATIC_PROVIDERS "all" CACHE STRING "")
add_subdirectory(rdma-core EXCLUDE_FROM_ALL)

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
  model.cc
  allocator.cc
  queue.cc
  cpu_allocator.cc
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
  fmt
  anl
)
target_include_directories(_C SYSTEM PRIVATE
  rdma-core
  ${CMAKE_CURRENT_BINARY_DIR}/rdma-core/include
  ${TorchPath}/include
  ${TorchPath}/include/torch/csrc/api/include
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(_C PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(_C PUBLIC ibverbs-static bnxt_re cxgb4 efa-static erdma hns irdma mana-static mlx4-static mlx5-static mthca ocrdma qedr vmw_pvrdma hfi1verbs ipathverbs rxe siw ibverbs-static)

target_compile_definitions(_C PRIVATE CUDADEVRT_PATH="${CUDADEVRT}")
