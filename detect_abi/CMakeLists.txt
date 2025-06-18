cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(detect_abi)

set(TorchPath "not-set" CACHE STRING "")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

add_subdirectory(../pybind11 pybind11)

set(ABI_VERSION "auto" CACHE STRING "The ABI version that pytorch was built with")
set(USE_CXX1_ABI "auto" CACHE STRING "_GLIBCXX_USE_CXX11_ABI option that pytorch was built with")

pybind11_add_module(moodist_detect_abi
  pybind.cc
)

target_link_libraries(moodist_detect_abi PUBLIC
  ${TorchPath}/lib/libtorch_python.so
)
target_include_directories(moodist_detect_abi SYSTEM PRIVATE
  ${TorchPath}/include
  ${TorchPath}/include/torch/csrc/api/include
)

target_compile_options(moodist_detect_abi PRIVATE -fabi-version=${ABI_VERSION})
target_compile_definitions(moodist_detect_abi PRIVATE _GLIBCXX_USE_CXX11_ABI=${USE_CXX1_ABI})
