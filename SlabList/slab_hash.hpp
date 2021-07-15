#ifndef SLAB_HASH_HPP
#define SLAB_HASH_HPP

#include <algorithm>
#include <CL/sycl.hpp>

#define WARP_SIZE 32
#define CONST 5

bool always_true();

template <typename T>
class slab_list {
public:
    slab_list(T em, sycl::queue &q);
    ~slab_list();

    void push_back(T el);
    void remove(int ind);
    int find(T el);
    T &get(int ind);

    T &operator[](int ind);
private:
    struct slab_node {
        slab_node() = default;
        slab_node(T em);
        int first_empty();

        T data[WARP_SIZE * CONST];
        slab_node *next = NULL;
        T empty;
    };

    void clear_rec(slab_node *node);

    slab_node *root;
    T empty;
    sycl::queue q_;
};

#include "slab_hash_impl.hpp"

#endif