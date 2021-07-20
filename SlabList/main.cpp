#include <iostream>
#include "slab_hash.hpp"
#include <random>
#include <map>
#include <chrono>

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

    sycl::nd_range<1> r = sycl::nd_range<1>(sycl::range<1> {8}, sycl::range<1> {8});

    /*int *a = sycl::malloc_shared<int>(1, q);
    int *expec = sycl::malloc_shared<int>(1, q);
    int *value = sycl::malloc_shared<int>(1, q);
    bool *flag = sycl::malloc_shared<bool>(1, q);
    *a = 33;
    *expec = 33;
    *value = 44;

    auto e =     q.parallel_for(r, [=] (sycl::nd_item<1> it) {
        *flag = sycl::ONEAPI::atomic_ref<int, sycl::ONEAPI::memory_order::acq_rel,
                                 sycl::ONEAPI::memory_scope::system, 
                                 sycl::access::address_space::global_space>(*a).compare_exchange_strong(*expec, *value);

    });
    e.wait();
    std::cout << *a << '\n';
    std::cout << *flag << '\n';*/


    slab_list<int, int> l(-1, q);
    /*
    int check;
    std::cout << rand() << '\n';
    for (int i = 0; i < WARP_SIZE * CONST * 2048; i++) {
        
        int ind = rand() % 30000;
        int val = rand() % 1000000;

        if(m.find(ind) != m.end()) continue;

        m[ind] = val;
        l.add(ind, val);

        if (i == 33 || i == 55 || i == 101 || i == 302) {
            check = ind;
        }
    }*/
/*
    auto begin = std::chrono::steady_clock::now();
    auto p = l.find(check);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Slab - " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << '\n';
    begin = std::chrono::steady_clock::now();
    int ans = m[check];
    m.find(check);
    end = std::chrono::steady_clock::now();

    std::cout << "Map - " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << '\n';

    ASSERT(ans == p.first, "map == slab");
    std::cout << p.second;*/
    l.add(1, 13);
    l.add(2, 13);
    l.add(3, 15);
    l.add(8, 11);
    l.add(6, 101);
    l.add(4, 22);
    l.add(5, 13);
    l.add(9, 13);
    l.add(10, 5);
    l.add(7, 11);
    l.add(11, 101);
    l.add(12, 22);
    l.add(20, 13);
    l.add(19, 13);
    l.add(18, 15);
    l.add(17, 11);
    l.add(16, 101);
    l.add(13, 22);
    l.add(14, 13);
    l.add(15, 13);
    l.add(21, 15);
    l.add(22, 11);
    l.add(23, 101);
    l.add(24, 22);
    l.add(30, 13);
    l.add(29, 13);
    l.add(28, 15);
    l.add(27, 11);
    l.add(26, 101);
    l.add(25, 22);
    l.add(31, 13);
    l.add(0, 13);

    l.add(1, 13);
    l.add(2, 13);
    l.add(3, 15);
    l.add(8, 11);
    l.add(6, 101);
    l.add(4, 22);
    l.add(5, 13);
    l.add(9, 13);
    l.add(10, 5);
    l.add(7, 11);
    l.add(11, 101);
    l.add(12, 22);
    l.add(20, 13);
    l.add(19, 13);
    l.add(18, 15);
    l.add(17, 11);
    l.add(16, 101);
    l.add(13, 22);
    l.add(14, 13);
    l.add(15, 13);
    l.add(21, 15);
    l.add(222, 11);
    l.add(23, 101);
    l.add(24, 22);
    l.add(30, 13);
    l.add(29, 13);
    l.add(28, 15);
    l.add(27, 11);
    l.add(26, 101);
    l.add(25, 22);
    l.add(31, 13);
    l.add(0, 13);

    ASSERT(l.find(0).first == 13, "zero check");
    ASSERT(l.find(0).second, "check zero second ar");
    ASSERT(l.find(23).first == 101, "23");
    ASSERT(l.find(222).first == 11, "two buckets test");
    
}
