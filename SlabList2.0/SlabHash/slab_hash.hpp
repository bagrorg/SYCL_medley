#pragma once

#include <CL/sycl.hpp>
#include <algorithm>

#define SUBGROUP_SIZE 16
#define CONST 64
#define SLAB_SIZE CONST * SUBGROUP_SIZE

#define CLUSTER_SIZE 1024

#define BUCKETS_COUNT 128

#define EMPTY_UINT32_T 4294967295

using std::pair;

template <size_t A, size_t B, size_t P>
struct Hasher {
    size_t operator()(const uint32_t &k) { return ((A * k + B) % P) % BUCKETS_COUNT; };
};

template <typename T>
struct SlabNode {
    SlabNode() = default;            
    SlabNode(T el) {
        for (int i = 0; i < SLAB_SIZE; i++) {
            data[i] = el;
        }
    }

    T data[SLAB_SIZE];
    sycl::global_ptr<SlabNode<T>> next = nullptr;
};

template <typename T>
struct SlabList {
    SlabList() = default;
    SlabList(sycl::queue &q, T empty) : _q(q) {
        root = sycl::global_ptr<SlabNode<T>>(sycl::malloc_shared<SlabNode<T>>(CLUSTER_SIZE, q));

        for (int i = 0; i < CLUSTER_SIZE - 1; i++) {
            *(root + i) = SlabNode(empty);
            (root + i)->next = (root + i + 1);
        }
    }
    ~SlabList() {
        //sycl::free(root, _q);
    }

    sycl::global_ptr<SlabNode<T>> root;
    sycl::queue _q;
};

template <typename K, typename T, typename Hash>
class SlabHash {
public:
    SlabHash() = default;
    SlabHash(K empty, Hash hasher, 
            sycl::global_ptr<SlabList<pair<K, T>>> lists,
            sycl::nd_item<1> & it, 
            sycl::global_ptr<SlabNode<pair<K, T>>> &iter, 
            const sycl::stream &out) : _lists(lists), _gr(it.get_group()), _it(it), 
                                       _empty(empty), _hasher(hasher), _iter(iter), 
                                       _out(out), _ind(_it.get_local_id()) { };

    void insert(K key, T val) {
        _key = key;
        _val = val;

        if (_ind == 0) {
            _iter = (_lists + _hasher(key))->root;
        }
        sycl::group_barrier(_gr);

        while (_iter != nullptr) {
            if (insert_in_node()) {
                break;
            } else if (_ind == 0) {
                _iter = _iter->next;
            }

            sycl::group_barrier(_gr);
        }
    }

    pair<T, bool> find(K key) {
        _key = key;
        _ans = {T(), false};
        
        if (_ind == 0) {
            _iter = (_lists + _hasher(key))->root;
        }
        sycl::group_barrier(_gr);

        while (_iter != nullptr) {
            if (find_in_node()) {
                break;
            } else if (_ind == 0) {
                _iter = _iter->next;
            }

            sycl::group_barrier(_gr);
        }

        return _ans;
    }

private:
    bool insert_in_node() {
        bool total_found = false;
        bool find = false;

        for(int i = _ind; i <= _ind + SUBGROUP_SIZE * (CONST - 1); i += SUBGROUP_SIZE) {
            find = ((_iter->data[i].first) == _empty);
            sycl::group_barrier(_gr);
            total_found = sycl::any_of_group(_gr, find);

            if (total_found) {
                if (insert_in_subgroup(find, i)) {
                    return true;
                }
            }
        }

        return false;
    }

    bool insert_in_subgroup(bool find, int i) {
        for (int j = 0; j < SUBGROUP_SIZE; j++) {
            if (sycl::group_broadcast(_gr, find, j)) {
                K tmp_empty = _empty;
                bool done = _ind == j ? sycl::ONEAPI::atomic_ref<K, sycl::ONEAPI::memory_order::acq_rel,
                                            sycl::ONEAPI::memory_scope::system,
                                            sycl::access::address_space::global_space>(
                                                _iter->data[i].first
                                            ).compare_exchange_strong(tmp_empty,
                                                                        _key) : false;
                if (done) {
                    _iter->data[i].second = _val;
                }
                if (sycl::group_broadcast(_gr, done, j)) {
                    return true;
                }
            }
        }

        return false;
    }

    bool find_in_node() {
        bool find = false;
        bool total_found = false;

        for(int i = _ind; i <= _ind + SUBGROUP_SIZE * (CONST - 1); i += SUBGROUP_SIZE) {
            find = ((_iter->data[i].first) == _key);
            sycl::group_barrier(_gr);
            total_found = sycl::any_of_group(_gr, find);
            
    
            if (total_found) {
                find_in_subgroup(find, i);
                return true;
            }
        }

        return false;
    }

    void find_in_subgroup(bool find, int i) {
        for (int j = 0; j < SUBGROUP_SIZE; j++) {
            if (sycl::group_broadcast(_gr, find, j)) {
                T tmp;
                if (_ind == j) tmp = _iter->data[i].second;                          //todo index shuffle

                _ans = {sycl::group_broadcast(_gr, tmp, j), true};
                break;
            }
        }
    }

    sycl::global_ptr<SlabList<pair<K, T>>> _lists;
    sycl::global_ptr<SlabNode<pair<K, T>>> &_iter;
    const sycl::stream &_out;
    sycl::group<1> _gr;
    sycl::nd_item<1> &_it;
    size_t _ind;

    K _empty;
    Hash _hasher;

    K _key;
    T _val;

    pair<T, bool> _ans;
};