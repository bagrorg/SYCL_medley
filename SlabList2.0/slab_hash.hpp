#pragma once

#include <CL/sycl.hpp>
#include <algorithm>

#define SUBGROUP_SIZE 8
#define CONST 1
#define SLAB_SIZE CONST * SUBGROUP_SIZE

#define CLUSTER_SIZE 1024

#define BUCKETS_COUNT 5

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
            sycl::nd_item<1> & it, sycl::global_ptr<SlabNode<pair<K, T>>> &iter, const sycl::stream &out) : _lists(lists), _gr(it.get_group()), _it(it), 
                                                                                  _empty(empty), _hasher(hasher), _iter(iter), _out(out) { };

    void insert(K key, T val) {
        auto ind = _it.get_local_id();
        //_out << ind << sycl::endl;
        
        bool total_found = false;
        bool find = false;

        if (ind == 0) _iter = (_lists + _hasher(key))->root;

        sycl::group_barrier(_gr);

        //_out << "Group - " << _it.get_group().get_id() <<
                //" - index - " << _it.get_local_id() << " - iter - " << key << ' ' << val << sycl::endl;

        while (_iter != nullptr) {
            //_out << ind << sycl::endl;
            for(int i = ind; i <= ind + SUBGROUP_SIZE * (CONST - 1); i += SUBGROUP_SIZE) {
                find = ((_iter->data[i].first) == _empty);
                //_out << "GR - " << _gr.get_id() << " ind " << ind << " " << find << sycl::endl;
                //waiting for all items
                if(key == 10) _out << "FIND - " << key << ' ' <<  ind << ' ' << find << sycl::endl;
                if (key == 10 && !find) _out << "ELEM - " << key << ' ' << ind << ' ' << (_iter->data[i].first) << ' ' << _empty << sycl::endl;
                sycl::group_barrier(_gr);
                total_found = sycl::any_of_group(_gr, find);
                //out << ind << ' ' << total_found << sycl::endl;
                if(key == 10) _out << "TOTAL - " << key << ' ' <<  ind << ' ' << total_found << sycl::endl;
                //If some item found something
                if (total_found) {
                    //if (ind == 0) _out << _gr.get_id() << sycl::endl;
                    total_found = false;
                    //Searching first item with find == 1                                                           TODO
                    for (int j = 0; j < SUBGROUP_SIZE; j++) {
                        //if item j has find == 1
                        if (sycl::group_broadcast(_gr, find, j)) {
                            //item j sets it to ans
                            K tmp_empty = _empty;
                            if(key == 10) _out << "TRY - " << key << ' ' <<  j << sycl::endl;

                            bool done = ind == j ? sycl::ONEAPI::atomic_ref<K, sycl::ONEAPI::memory_order::acq_rel,
                                                        sycl::ONEAPI::memory_scope::system,
                                                        sycl::access::address_space::global_space>(
                                                            _iter->data[i].first
                                                        ).compare_exchange_strong(tmp_empty,
                                                                                    key) : false;
                            if (key == 10) _out << "POSSIBLE SOOQA - " << _empty << ' ' << _iter->data[i].first << ' ' << sycl::endl;
                            if (done) {
                                //out << "S - " << ind << sycl::endl;
                                //_out << "GR - " << _gr.get_id() << ", ind -" << ind << ' ' << key << sycl::endl;
                                _iter->data[i].second = val;
                            }
                            if (sycl::group_broadcast(_gr, done, j)) {
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
            if (total_found) {
                break;
            }
            //0st item jumping to another node
            if(key == 10)_out << "PREGO - " << key << ' ' << ind << sycl::endl;
            if (ind == 0) {
                if(key == 10)_out << "GO - " << key << sycl::endl;
                _iter = _iter->next;
            }
            if(key == 10)_out << "NOW - " << key << ' ' << ind << ' ' << _iter << sycl::endl;
            //waiting for 0st item jump and all others
            sycl::group_barrier(_gr);
        }
    }

    pair<T, bool> find(K key) {
        pair<T, bool> ans = {T(), false};
        //Index inside work-group (we have only one wg so it is equals to global id)
        auto ind = _it.get_local_id();
        

        //Flags
        bool find = false;          //Item found element with key similar to accKey
        bool total_found = false;   //Any of item found something

        while (_iter != NULL) {
            //                         FOR LOOP EXPLANATION
            //[. . . . . . . |. . . . . . . |. . . . . . . |. . . . . . . |...]
            //   ^          W_S                ^
            //   |                             |
            //Locally : ind                Locally : ind
            //Globally : ind        Globally : ind + WARP_SIZE * 2
            for(int i = ind; i <= ind + SUBGROUP_SIZE * (CONST - 1); i += SUBGROUP_SIZE) {
                find = ((_iter->data[i].first) == key);
                //waiting for all items
                sycl::group_barrier(_gr);
                total_found = sycl::any_of_group(_gr, find);
                
                //If some item found something
                if (total_found) {
                    //Searching first item with find == 1                                                           TODO
                    for (int j = 0; j < SUBGROUP_SIZE; j++) {
                        //if item j has find == 1
                        if (sycl::group_broadcast(_gr, find, j)) {
                            //item j sets it to ans
                            T tmp;
                            if (ind == j) tmp = _iter->data[i].second;

                            ans = {sycl::group_broadcast(_gr, tmp, j), true};
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
            if (ind == 0) _iter = _iter->next;
            //waiting for 0st item jump and all others
            sycl::group_barrier(_gr);
        }

        return ans;
    }

private:
    sycl::global_ptr<SlabList<pair<K, T>>> _lists;
    sycl::global_ptr<SlabNode<pair<K, T>>> &_iter;
    const sycl::stream &_out;
    sycl::group<1> _gr;
    sycl::nd_item<1> &_it;

    K _empty;
    Hash _hasher;
};