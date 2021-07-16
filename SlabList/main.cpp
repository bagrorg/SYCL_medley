#include <iostream>
#include "slab_hash.hpp"
#include <random>
#include <map>

#define ASSERT(condition, name) \
            if(!(condition)) { \
                std::cerr << name << ' ' << "FAILED\n"; \
            } else { \
                std::cout << name << ' ' << "DONE\n"; \
            }


int main() {
    srand(time(NULL));
    std::map<int, int> m;
    sycl::queue q {sycl::gpu_selector() } ;
    std::cout << q.get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Subgroup sizes - ";
    for(auto e: q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
        std::cout << e << ' ';
    }
    std::cout << "\n-------------\n\n";

    slab_list<int, int> l(-1, q);
    int check;

    for (int i = 0; i < WARP_SIZE * CONST * 2048; i++) {
        int ind = rand() % 30000;
        int val = rand() % 1000000;

        if(m.find(ind) != m.end()) continue;

        m[ind] = val;
        l.add(ind, val);

        if (i == 33 || i == 55 || i == 101 || i == 302) {
            check = ind;
        }
    }

    auto p = l.find(check);

    int ans = m[check];


    std::cout << (ans == p.first);
}
