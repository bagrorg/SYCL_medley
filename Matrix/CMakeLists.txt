set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_FLAGS "-fsycl -O2") 
 
project(SYCL_Matrix)

add_executable(${CMAKE_PROJECT_NAME} matrix.cpp)