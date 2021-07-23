set(slab_hash_src
    slab_hash.hpp
    slab_hash.cpp
)

add_library(slab_hash ${slab_hash_src})
target_include_directories(slab_hash PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(slab_hash 
    PRIVATE sycl
)
target_compile_options(slab_hash PRIVATE -fsycl)
target_link_options(slab_hash PRIVATE -fsycl)
