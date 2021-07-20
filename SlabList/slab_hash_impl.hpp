#ifndef SLAB_HASH_IMPL_HPP
#define SLAB_HASH_IMPL_HPP


#include "slab_hash.hpp"
#include <strings.h>

template <typename K, typename T>
slab_list<K, T>::slab_list(K em, sycl::queue &q) : empty(em), q_(q) {
    cluster = sycl::malloc_shared<slab_node>(CLUSTER_SIZE, q_);
    //std::cout << empty << std::endl;
    for(int i = 0; i < CLUSTER_SIZE; i++) {
        cluster[i] = slab_node(empty);
    }

    root = &cluster[0];
}

template <typename K, typename T>
slab_list<K, T>::~slab_list() {
    clear_rec(root);

}

template <typename K, typename T>
void slab_list<K, T>::clear_rec(slab_node *node) {
    sycl::free(cluster, q_);
}

template <typename K, typename T>
void slab_list<K, T>::add(K key, T el) {
    sycl::nd_range<1> r { WARP_SIZE, WARP_SIZE};
    auto iter = &cluster[0];                                                           //TODO если нет проблем с захватом то исправить
    slab_node* prev = NULL;
    //std::cout << "HERE" << std::endl;

    {
        auto buffIter = sycl::buffer(&iter, sycl::range(1));
        auto buffNumOfBuck = sycl::buffer(&num_of_buckets, sycl::range(1));
        auto buffer_em = sycl::buffer(&empty, sycl::range(1));
        auto buffKey = sycl::buffer(&key, sycl::range(1));
        auto buffEl = sycl::buffer(&el, sycl::range(1));
        auto buffPrev = sycl::buffer(&prev, sycl::range(1));
        auto buffClust = sycl::buffer(&cluster, sycl::range(1));

      //std::cout << "HERE" << std::endl;

        q_.submit([&](sycl::handler &cgh) {
            sycl::stream out(10000, 100, cgh);

            auto accIter = sycl::accessor{buffIter, cgh, sycl::read_write};
            auto accEm = sycl::accessor{buffer_em, cgh, sycl::read_write};
            auto accKey = sycl::accessor{buffKey, cgh, sycl::read_only};
            auto accEl = sycl::accessor{buffEl, cgh, sycl::read_only};
            auto accPrev = sycl::accessor{buffPrev, cgh, sycl::read_write};
            auto accClust = sycl::accessor{buffClust, cgh, sycl::read_write};
            auto accNumOfBuck = sycl::accessor{buffNumOfBuck, cgh, sycl::read_write};

            //std::cout << "HERE" << std::endl;
            

            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                auto ind = it.get_local_id();
                auto sg = it.get_sub_group();
                bool total_found = false;
                bool find = false;

                while(1) {
                    while ((*accIter.get_pointer()) != NULL) {
                        for(int i = ind; i <= ind + WARP_SIZE * (CONST - 1); i += WARP_SIZE) {
                            find = (((*accIter.get_pointer())->data[i].first) == *(accEm.get_pointer()));
                            //out << ind << ' ' << find << ' ' << *(accEm.get_pointer()) << ' ' << ((*accIter.get_pointer())->data[i].first) << sycl::endl;
                            //waiting for all items
                            sg.barrier();
                            total_found = sycl::any_of_group(sg, find);
                            //out << ind << ' ' << total_found << sycl::endl;
                            
                            //If some item found something
                            if (total_found) {
                                total_found = false;
                                //Searching first item with find == 1                                                           TODO
                                for (int j = 0; j < WARP_SIZE; j++) {
                                    //if item j has find == 1
                                    if (sycl::group_broadcast(sg, find, j)) {
                                        //item j sets it to ans
                                        //out << j << sycl::endl;
                                        bool done = ind == j ? sycl::ONEAPI::atomic_ref<K, sycl::ONEAPI::memory_order::acq_rel,
                                                                    sycl::ONEAPI::memory_scope::system,
                                                                    sycl::access::address_space::global_space>(
                                                                        ((*accIter.get_pointer())->data[i].first)
                                                                    ).compare_exchange_strong(*accEm.get_pointer(),
                                                                                                *accKey.get_pointer()) : false;
                                        if (done) {
                                            //out << "S - " << ind << sycl::endl;
                                            (*accIter.get_pointer())->data[i].second = *accEl.get_pointer();
                                        }
                                        if (sycl::group_broadcast(sg, done, j)) {
                                            //out << ind << "here" << sycl::endl;
                                            total_found = true;
                                            break;
                                        }
                                    }
                                }
                                //breaking that job is done
                                if (total_found) break;
                            }
                        }
                        if (total_found) break;
                        //0st item jumping to another node
                        if (ind == 0) {
                            (*accPrev.get_pointer()) = (*accIter.get_pointer());
                            (*accIter.get_pointer()) = (*accIter.get_pointer())->next;
                        }
                        //waiting for 0st item jump and all others
                        sg.barrier();
                    }
                    if (total_found) break;
                    else {
                        //out << ind << sycl::endl;
                        if (ind == 0) {
                            
                            (*accPrev.get_pointer())->next = &(*accClust.get_pointer())[*accNumOfBuck.get_pointer()];
                            (*accIter.get_pointer()) = (*accPrev.get_pointer())->next;
                            *accNumOfBuck.get_pointer() += 1;
                        }
                        //out << *accIter.get_pointer() << sycl::endl;
                        sg.barrier();
                    }
                    
                }
            });

            q_.wait();
        });
    }
}


template <typename K, typename T>
pair<T, bool> slab_list<K, T>::find(const K& key) {
    //Work size is equals to hardware sub-group
    sycl::nd_range<1> r { WARP_SIZE, WARP_SIZE};
    //By default we didnt find anything                                                                                  TODO: std::optional
    pair<T, bool> ans = {T(), false};
    //Our iterator
    auto iter = cluster;

    {
        //Taking ans pair, key and iterator on board
        auto buffAns = sycl::buffer(&ans, sycl::range(1));
        auto buffKey = sycl::buffer(&key, sycl::range(1));
        auto buffIter = sycl::buffer(&iter, sycl::range(1));

        q_.submit([&] (sycl::handler &cgh) {
            //Accessing buffers
            auto accAns = sycl::accessor{buffAns, cgh, sycl::write_only};
            auto accKey = sycl::accessor{buffKey, cgh, sycl::read_only};
            auto accIter = sycl::accessor{buffIter, cgh, sycl::read_write};

            //DEBUG
            sycl::stream out(1020004, 10000, cgh);
            

            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                //Index inside work-group (we have only one wg so it is equals to global id)
                auto ind = it.get_local_id();
                auto sg = it.get_sub_group();

                //Flags
                bool find = false;          //Item found element with key similar to accKey
                bool total_found = false;   //Any of item found something

                while ((*accIter.get_pointer()) != NULL) {
                    //                         FOR LOOP EXPLANATION
                    //[. . . . . . . |. . . . . . . |. . . . . . . |. . . . . . . |...]
                    //   ^          W_S                ^
                    //   |                             |
                    //Locally : ind                Locally : ind
                    //Globally : ind        Globally : ind + WARP_SIZE * 2
                    for(int i = ind; i <= ind + WARP_SIZE * (CONST - 1); i += WARP_SIZE) {
                        find = (((*accIter.get_pointer())->data[i].first) == (*accKey.get_pointer()));
                        //waiting for all items
                        sg.barrier();
                        total_found = sycl::any_of_group(sg, find);
                        
                        //If some item found something
                        if (total_found) {
                            //Searching first item with find == 1                                                           TODO
                            for (int j = 0; j < WARP_SIZE; j++) {
                                //if item j has find == 1
                                if (sycl::group_broadcast(sg, find, j)) {
                                    //item j sets it to ans
                                    if (ind == j) *accAns.get_pointer() = {(*accIter.get_pointer())->data[i].second, true};
                                    //all items breaking
                                    break;
                                }
                            }
                            //breaking that job is done
                            break;
                        }
                    }
                    if (total_found) break;
                    //0st item jumping to another node
                    if (ind == 0) (*accIter.get_pointer()) = (*accIter.get_pointer())->next;
                    //waiting for 0st item jump and all others
                    sg.barrier();
                }
            });
            q_.wait();
        });
    }
    return ans;
}


template <typename K, typename T>
slab_list<K, T>::slab_node::slab_node(K em) {
    for (int i = 0; i < WARP_SIZE * CONST; i++) {
        data[i].first = em;
        data[i].second = T();
    }
}

#endif
