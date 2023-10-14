cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(moodist)

project(moodist LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -ftemplate-backtrace-limit=0 -Bsymbolic -ffast-math -mavx -mavx2 -msse -msse2 -msse3 -msse4.1 -mpopcnt -mcx16")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

execute_process(
    COMMAND python -c "import os, torch; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

message(STATUS "PyTorch compilation flags: ${TORCH_CXX_FLAGS}")
message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(CUDAToolkit)

message(STATUS "CUDA includes are ${CUDAToolkit_INCLUDE_DIRS}")


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
)
target_link_libraries(
  _C
  PRIVATE
  ${TORCH_LIBRARIES}
  ${TorchPath}/lib/libtorch_python${CMAKE_SHARED_LIBRARY_SUFFIX}
  Threads::Threads
  fmt
  anl numa ${FABRIC_LIBRARY}
  CUDA::cudart CUDA::nvml
)
target_include_directories(_C SYSTEM PRIVATE rdma-core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(_C PRIVATE ${TORCH_CXX_FLAGS})

target_link_libraries(_C PUBLIC ibverbs-static bnxt_re cxgb4 efa-static erdma hns irdma mana-static mlx4-static mlx5-static mthca ocrdma qedr vmw_pvrdma hfi1verbs ipathverbs rxe siw ibverbs-static)


