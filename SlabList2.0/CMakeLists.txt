cmake_minimum_required(VERSION 3.10)
include(FetchContent)

set(CMAKE_CXX_COMPILER "/home/bagrorg/sycl_workspace/llvm/build/bin/clang++")
set(CMAKE_CXX_FLAGS "-fsycl")
# set the project name
project(medley_slab)

add_subdirectory(SlabHash)

# add the executable
add_executable(medley_slab main.cpp)
target_link_libraries(${PROJECT_NAME} slab_hash)


option(ENABLE_TESTS ON)
if(ENABLE_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  
  add_subdirectory(tests)
endif()