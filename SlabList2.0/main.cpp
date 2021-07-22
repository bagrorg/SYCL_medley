#include <CL/sycl.hpp>
#include "slab_hash.hpp"

int main() {
    auto hn = [] (sycl::exception_list el) {
        for (auto &e: el) {
            try{std::rethrow_exception(e);}
            catch(sycl::exception &e) {
                std::cout << "ASYNC EX\n";
                std::cout << e.what() << '\n';
            }
        }
    };

    sycl::queue q{ sycl::gpu_selector(), hn };
    std::cout << q.get_device().get_info<sycl::info::device::name>() << '\n';
    std::cout << "Subgroup sizes - ";
    for(auto e: q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
        std::cout << e << ' ';
    }
    std::cout << "\n-------------\n\n";

    sycl::nd_range<1> r{SUBGROUP_SIZE * 1024, SUBGROUP_SIZE};

    std::vector<SlabList<pair<uint32_t, uint32_t>>> lists(BUCKETS_COUNT);
    for(auto &e: lists) {
        e = SlabList<pair<uint32_t, uint32_t>>(q, {EMPTY_UINT32_T, 0});
    }

    


    {
        sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
        sycl::buffer<sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(1024);

        q.submit([&](sycl::handler &cgh) {
            auto l = sycl::accessor(ls, cgh, sycl::read_write);
            auto itrs = sycl::accessor(its, cgh, sycl::read_write);
            sycl::stream out(10000000, 10000, cgh);

            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                //sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>> iter;
                //out << it.get_group(). << ' ' << it.get_sub_group().get_local_id() << ' ' << it.get_local_id() << sycl::endl;
                //if( it.get_global_id() == 5) out << it.get_group().get_id() << ' ' << it.get_local_id() << sycl::endl;
                    Hasher<13, 24, 79> h;
                    SlabHash<uint32_t, uint32_t, Hasher<13, 24, 79>> ht(EMPTY_UINT32_T, h, l.get_pointer(), it, itrs[it.get_group().get_id()], out);

                    ht.insert(it.get_group().get_id(), 12);
            });
                
            q.wait();
                //out << gr_ind << sycl::endl;
        }).wait();

    }

    {
        //sycl::buffer<bool> ok(1);
        sycl::buffer<SlabList<pair<uint32_t, uint32_t>>> ls(lists);
        sycl::buffer<sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>>> its(1024);

        q.submit([&](sycl::handler &cgh) {
            //auto accok = sycl::accessor(ok, cgh, sycl::read_write);
            auto l = sycl::accessor(ls, cgh, sycl::read_write);
            auto itrs = sycl::accessor(its, cgh, sycl::read_write);
            sycl::stream out(100000, 1000, cgh);

            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                //sycl::global_ptr<SlabNode<pair<uint32_t, uint32_t>>> iter;
                //out << it.get_group(). << ' ' << it.get_sub_group().get_local_id() << ' ' << it.get_local_id() << sycl::endl;
                //if( it.get_global_id() == 5) out << it.get_group().get_id() << ' ' << it.get_local_id() << sycl::endl;
                    Hasher<13, 24, 79> h;
                    SlabHash<uint32_t, uint32_t, Hasher<13, 24, 79>> ht(EMPTY_UINT32_T, h, l.get_pointer(), it, itrs[it.get_group().get_id()], out);

                    if(!ht.find(it.get_group().get_id()).second) {
                        out << "ERROR" << sycl::endl;
                    }
            });
            
            //std::cout << *accok.get_pointer() << std::endl;
                
                //out << gr_ind << sycl::endl;
        }).wait();

    }

    std::ofstream out("out.txt");
    for(auto &e: lists) {
        out << "LIST - ";
        for(int i = 0; i < CLUSTER_SIZE; i++) {
            out << (e.root + i) << " -- " ;
            for(int j = 0; j < SLAB_SIZE; j++) {
                out << ((*(e.root + i)).data[j].first) << ' ';
            }
            out << "\n       ";
        }
        out << '\n';
    }

    std::cout << "cum" << std::endl;
}