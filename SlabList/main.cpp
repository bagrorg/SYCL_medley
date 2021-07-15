#include <iostream>
#include "slab_hash.hpp"

#define ASSERT(condition, name) \
            if(!(condition)) { \
                std::cerr << name << ' ' << "FAILED\n"; \
            } else { \
                std::cout << name << ' ' << "DONE\n"; \
            }


int main() {
    sycl::queue q {sycl::gpu_selector() } ;
    std::cout << q.get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Subgroup sizes - ";
    for(auto e: q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
        std::cout << e << ' ';
    }
    std::cout << "\n-------------\n\n";
    

    slab_list<int> l(-1, q);

    for (int i = 0; i < WARP_SIZE * CONST * 4; i++) {
        l.push_back(i % (WARP_SIZE * CONST));
    }

    ASSERT(l[5] == l[165], "1");
    ASSERT(l[167] == 7, "2");

    l.remove(5);
    ASSERT(l[5] != l[165] && l[5] == -1, "3");

    ASSERT(l.find(5) == 165, "4");
}