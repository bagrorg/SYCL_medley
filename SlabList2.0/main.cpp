#include <CL/sycl.hpp>
#include "SlabHash/slab_hash.hpp"


#define WORKSIZE 10000

int main() {
    sycl::queue q{ sycl::gpu_selector() };
    sycl::nd_range<1> r{SUBGROUP_SIZE * 3, SUBGROUP_SIZE};

    std::vector<SlabList<pair<uint32_t, uint32_t>>> lists(BUCKETS_COUNT);
    for(auto &e: lists) {
        e = SlabList<pair<uint32_t, uint32_t>>(q, {EMPTY_UINT32_T, 0});
    }

    std::vector<pair<uint32_t, uint32_t>> testUniv = { {1, 2},
                                                       {5, 2},
                                                       {101, 3},
                                                       {10932, 5},
                                                       {3, 0},
                                                       {10, 10} };
    
    Hasher<13, 24, 343> h;

    for(auto &e: testUniv) {
        auto r = lists[h(e.first)].root;

        for(int i = 0; i < SLAB_SIZE; i++) {
            if (r->data[i].first == EMPTY_UINT32_T) {
                r->data[i] = e;
                break;
            }
        }
    }
    std::vector<pair<bool, bool>> checks(6);
    {
        sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
        sycl::buffer<sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(3);
        sycl::buffer<pair<uint32_t, uint32_t>> buffTestUniv(testUniv);
        sycl::buffer<pair<bool, bool>> buffChecks(checks);

        q.submit([&](sycl::handler &cgh) {
            auto l = sycl::accessor(ls, cgh, sycl::read_write);
            auto itrs = sycl::accessor(its, cgh, sycl::read_write);
            auto tests = sycl::accessor(buffTestUniv, cgh, sycl::read_only);
            auto accChecks = sycl::accessor(buffChecks, cgh, sycl::write_only);
            sycl::stream out(100000, 1000, cgh);

            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                size_t ind = it.get_group().get_id();
                Hasher<13, 24, 343> h;
                SlabHash<uint32_t, uint32_t, Hasher<13, 24, 343>> ht(EMPTY_UINT32_T, 
                                                                    h, l.get_pointer(), it, 
                                                                    itrs[it.get_group().get_id()], out);

                for (int i = ind * 2; i < ind * 2 + 2; i++) {
                    auto ans = ht.find(tests[i].first);
                    
                    accChecks[i] = { ans.second, ans.first == tests[i].second };
                }
            });
        }).wait();
    }

    for(auto &e: checks) {
        std::cout << e.first << ' ' << e.second << std::endl;
    }
}