#ifndef SLAB_HASH_IMPL_HPP
#define SLAB_HASH_IMPL_HPP


#include "slab_hash.hpp"

template <typename T>
slab_list<T>::slab_list(T em, sycl::queue &q) : empty(em), q_(q) {
    root = new slab_node(empty);
}

template <typename T>
slab_list<T>::~slab_list() {
    clear_rec(root);
}

template <typename T>
void slab_list<T>::clear_rec(slab_node *node) {
    if (node->next != NULL) clear_rec(node->next);
    delete node;
}

template <typename T>
void slab_list<T>::push_back(T el) {
    int ind = find(empty);
    if (ind == -1) {
        slab_node *it = root;
        while (it->next != NULL) it = it->next;
        it->next = new slab_node(empty);

        it->next->data[0] = el;
        return;
    }

    get(ind) = el;
}

template <typename T>
void slab_list<T>::remove(int ind) {
   get(ind) = empty;
}

template <typename T>
T &slab_list<T>::operator[](int ind) {
    return get(ind);
}

template <typename T>
int slab_list<T>::find(T el) {
    slab_node *it = root;
    int ind = 0;

    while(it != NULL) {
        for (int i = 0; i < WARP_SIZE * CONST; i++) {
            if (it->data[i] == el) return ind + i;
        }

        ind += WARP_SIZE * CONST;
        it = it->next;
    }

    return -1;
}

template <typename T>
T &slab_list<T>::get(int ind) {
    slab_node *it = root;
    int range = WARP_SIZE * CONST;
    
    while (ind >= range) {
        it = it->next;
        range += WARP_SIZE * CONST;
    }
    
    ind = ind % (WARP_SIZE * CONST);
    return it->data[ind];
}

template <typename T>
slab_list<T>::slab_node::slab_node(T em) : empty(em) {
    for (int i = 0; i < WARP_SIZE * CONST; i++) {
        data[i] = em;
    }
}

template <typename T>
int slab_list<T>::slab_node::first_empty() {
    for (int i = 0; i < WARP_SIZE * CONST; i++) {
        if (data[i] == empty) return i;
    }

    return -1;
}

#endif