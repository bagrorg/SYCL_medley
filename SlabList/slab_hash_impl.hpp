#ifndef SLAB_HASH_IMPL_HPP
#define SLAB_HASH_IMPL_HPP


#include "slab_hash.hpp"
#include <CL/sycl/atomic.hpp>

template <typename K, typename T>
slab_list<K, T>::slab_list(K em, sycl::queue &q) : empty(em), q_(q) {
    cluster = sycl::malloc_shared<slab_node>(CLUSTER_SIZE, q_);
    cluster[0] = slab_node();
    root = &cluster[0];
    root->set_slab_node(em);
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
    auto iter = cluster;

    while (iter != NULL) {
        for (int i = 0; i < WARP_SIZE * CONST; i++) {
            if (iter->data[i].first == empty) {
                iter->data[i] = {key, el};
                return;
            }
        }
        if (iter->next == NULL) break;
        iter = iter->next;
    }
    
    if (num_of_buckets == CLUSTER_SIZE) return;

    cluster[num_of_buckets] = slab_node();
    cluster[num_of_buckets].set_slab_node(empty);

    iter->next = &cluster[num_of_buckets];
    num_of_buckets++;

    iter->next->data[0] = {key, el};
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
void slab_list<K, T>::slab_node::set_slab_node(K em) {
    for (int i = 0; i < WARP_SIZE * CONST; i++) {
        data[i].first = em;
    }
}

#endif
