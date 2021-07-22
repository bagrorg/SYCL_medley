cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "/home/bagrorg/sycl_workspace/llvm/build/bin/clang++")
set(CMAKE_CXX_FLAGS "-fsycl")
# set the project name
project(slab_hash)

# add the executable
add_executable(slab_hash main.cpp slab_hash.hpp)