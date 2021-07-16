#ifndef SLAB_HASH_IMPL_HPP
#define SLAB_HASH_IMPL_HPP


#include "slab_hash.hpp"
#include <CL/sycl/atomic.hpp>

template <typename K, typename T>
slab_list<K, T>::slab_list(K em, sycl::queue &q) : empty(em), q_(q) {
    cluster = sycl::malloc_shared<slab_node>(CLUSTER_SIZE, q_);
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

    iter->next = &cluster[num_of_buckets];
    num_of_buckets++;

    iter->next->data[0] = {key, el};
}


template <typename K, typename T>
pair<T, bool> slab_list<K, T>::find(const K& key) {
    sycl::nd_range<1> r { WARP_SIZE, WARP_SIZE};
    pair<T, bool> ans = {T(), false};
    auto iter = cluster;

    {
        auto buffAns = sycl::buffer(&ans, sycl::range(1));
        auto buffKey = sycl::buffer(&key, sycl::range(1));
        auto buffIter = sycl::buffer(&iter, sycl::range(1));

        q_.submit([&] (sycl::handler &cgh) {
            auto accAns = sycl::accessor{buffAns, cgh, sycl::write_only};
            auto accKey = sycl::accessor{buffKey, cgh, sycl::read_only};
            sycl::accessor<slab_list<int, int>::slab_node *, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> accIter = sycl::accessor{buffIter, cgh, sycl::read_write};
            sycl::stream out(1024, 10, cgh);
            cgh.parallel_for(r, [=](sycl::nd_item<1> it) {
                auto ind = it.get_local_id();
                bool find = false;
                 while ((*accIter.get_pointer()) != NULL) {
                     for(int i = ind; i <= ind + WARP_SIZE * (CONST - 1); i += WARP_SIZE) {
                         find = ((*accIter.get_pointer())->data[ind].first == *accKey.get_pointer());
                        if (find) *accAns.get_pointer() = {(*accIter.get_pointer())->data[ind].second, true};
                     }
                     if (sycl::any_of_group(it.get_group(), find)) {
                         break;
                    }
                    (*accIter.get_pointer()) = (*accIter.get_pointer())->next;
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
