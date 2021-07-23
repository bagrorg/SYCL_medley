enable_testing()

add_executable(slab_tests slab_tests.cpp)

target_include_directories(slab_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(slab_tests slab_hash GTest::gtest)

target_compile_options(slab_tests PRIVATE -fsycl)
target_link_options(slab_tests PRIVATE -fsycl)

add_test(slab_tests slab_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)