#ifndef SLAB_HASH_HPP
#define SLAB_HASH_HPP

#include <algorithm>
#include <CL/sycl.hpp>

#define WARP_SIZE 8
#define CONST 5
#define CLUSTER_SIZE 1024

using std::pair;

template <typename K, typename T>
class slab_list {
public:
    slab_list(K em, sycl::queue &q);
    ~slab_list();

    void add(K key, T el);
    pair<T, bool> find(const K& key);
private:
    struct slab_node {
        slab_node() = default;                          //TODO convert set slab node to constructor
        void set_slab_node(K em);

        pair<K, T> data[WARP_SIZE * CONST];
        slab_node *next = NULL;
        K empty;
    };

    void add_bucket();
    void clear_rec(slab_node *node);

    slab_node* cluster;

    slab_node *root;
    K empty;
    size_t num_of_buckets = 1;
    sycl::queue q_;
};

#include "slab_hash_impl.hpp"

#endif
