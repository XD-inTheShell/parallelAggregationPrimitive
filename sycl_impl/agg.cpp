#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <unordered_map> 
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include "../common.h"

#include <sycl/sycl.hpp>
#include <sycl/atomic.hpp>

#define kHashTableCapacity 128//16384
#define N 1000000
#define KEY_EMPTY 0xFFFFFFFF


struct kv
{
    uint32_t key;
    uint32_t value;
};


int readFile(std::string fileName, std::vector<uint32_t> &keys, std::vector<Value> &values, int size){
    std::ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        return false;
    }
    std::string line;
    int i = 0;
    while (std::getline(inFile, line)) {
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        int key         = (int)atoi(str.c_str());
        keys.push_back(key);
        std::getline(sstream, str, ' ');
        #ifdef VALUEINT
            Value value    = (int)atoi(str.c_str());
        #else
            Value value    = (double)atof(str.c_str());
        #endif
        values.push_back(value);
        i++;
        if(i == size) break;
    }
    inFile.close();
    return 0;
}

int writeFile(std::string fileName, std::vector<int> &keys, std::vector<Value> &values, std::unordered_map<int, Value> &umap){
    std::map<int, Value> res(umap.begin(), umap.end());

    fileName = "out.txt";
    std::ofstream file(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return 0;
    }
    file << std::setprecision(9);
    for (auto d : res) {
        file << d.first << " " << d.second << std::endl;
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    // std::ofstream file(fileName);
    // if (!file) {
    //     std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    //     return 0;
    // }
    // file << std::setprecision(10);
    // for (long unsigned int i=0; i<keys.size();i++) {
    //     file << keys[i] << " " << values[i] << std::endl;
    // }
    // file.close();
    // if (!file)
    //     std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    return true;
}

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

using namespace sycl;

//hash and hashtable functions from https://github.com/nosferalatu/SimpleGPUHashTable
inline uint32_t hash(uint32_t k){
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
    // return 0;
}

// void atomic_update(uint32_t* value, unint32_t*)

void insert(kv* hashtable, uint32_t key, uint32_t value)
{
    uint32_t slot = hash(key);

    while (true)
    {
        //uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        // uint32_t prev = hashtable[slot].key;
        // if(prev == KEY_EMPTY){
        //     hashtable[slot].key = key;
        // }
        auto atm = atomic_ref<uint32_t, 
                    memory_order::relaxed,
                    memory_scope::device,
                    access::address_space::global_space>(hashtable[slot].key);
        uint32_t expected = KEY_EMPTY;
        uint32_t desired = key;
        atm.compare_exchange_strong(expected, desired);
        if (expected == KEY_EMPTY)
        {   
            //os << "insert of" << key << " "<< value<< "\n";
            auto ref = atomic_ref<
                uint32_t, 
                memory_order::relaxed,
                memory_scope::device,
                access::address_space::global_space>(hashtable[slot].value);
            ref.fetch_add(value);
            break;
        }
        else if(expected == key){
            //os << "update of" << key << " "<< value<< "\n";
            auto ref = atomic_ref<
                uint32_t, 
                memory_order::relaxed,
                memory_scope::device,
                access::address_space::global_space>(hashtable[slot].value);
            ref.fetch_add(value);
            //os << "updated of" << key << " "<< hashtable[slot].value<< "\n";
            break;
        }
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}

uint32_t lookup(kv* hashtable, uint32_t key)
{
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                return slot;
            }
            if (hashtable[slot].key == KEY_EMPTY)
            {
                return KEY_EMPTY;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
}



int main(){
    //auto selector = default_selector_v;
    auto selector = cpu_selector_v;

    queue q(selector, exception_handler);

    std::cerr << "Running on device: " << q.get_device().get_info<info::device::name>() << std::endl;

    std::cerr << "hi" << std::endl;
    std::vector<uint32_t> vec_keys;
    std::vector<uint32_t> vec_values;
    readFile("../testcases/inputs/in.txt", vec_keys, vec_values, N);
    uint32_t keys[N];
    uint32_t values[N];
    std::copy(vec_keys.begin(), vec_keys.end(), keys);
    std::copy(vec_values.begin(), vec_values.end(), values);

    uint32_t *dev_keys;
    uint32_t *dev_vals;
    kv *hashtable;
    kv *hashtable_host;

    dev_keys = sycl::malloc_device<uint32_t>(N, q);
    dev_vals = sycl::malloc_device<uint32_t>(N, q);
    hashtable = sycl::malloc_device<kv>(kHashTableCapacity, q);
    hashtable_host = sycl::malloc_host<kv>(kHashTableCapacity, q);

    
    //q.memset(hashtable, 0xFF, kHashTableCapacity*sizeof(kv));
    q.memcpy(dev_keys, &keys, sizeof(uint32_t)*N).wait();
    q.memcpy(dev_vals, &values, sizeof(uint32_t)*N).wait();

    q.submit([&](handler& h){
        h.parallel_for(kHashTableCapacity, [=](item<1> i){
            hashtable[i].key = KEY_EMPTY;
            hashtable[i].value = 0;
        });
    });
    
    // q.memcpy(&values, dev_vals, values.size()*sizeof(uint32_t)).wait();
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    q.submit([&](handler& h){
        //stream os(1024, 128, h);
        h.parallel_for(N, [=](item<1> i){
                uint32_t key = dev_keys[i];
                uint32_t value = dev_vals[i];
                uint32_t slot = lookup(hashtable, key);
                if (slot != KEY_EMPTY){
                    auto ref = atomic_ref<
                        uint32_t, 
                        memory_order::relaxed,
                        memory_scope::device,
                        access::address_space::global_space>(hashtable[slot].value);
                    ref.fetch_add(value);
                }
                else{
                    insert(hashtable, key, value);
                }
            //}
            //os << "done" << "\n";
        });
    }).wait();//.wait();
    auto t1 = std::chrono::steady_clock::now();
    q.wait();
    q.memcpy(hashtable_host, hashtable, sizeof(kv)*kHashTableCapacity).wait();
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> submission_time = t1 - t0;
    std::chrono::duration<double> total_time = t2 - t0;
    std::chrono::duration<double> diff = t2-t1;

    std::cout << "submission time " << submission_time.count()*1000 << "ms" << "\n";
    std::cout << "total time " << total_time.count()*1000 << "ms" << "\n";
    std::cout << "diff" << diff.count()*1000 << "ms\n";

    
    std::map<uint32_t, uint32_t> res;
    for(int i = 0; i < kHashTableCapacity; i++){
        if(hashtable_host[i].key != KEY_EMPTY){
            //std::cerr << hashtable_host[i].key << "  " << hashtable_host[i].value << "\n";
            res.insert({hashtable_host[i].key, hashtable_host[i].value});
        }
    }

    std::string fileName = "out.txt";
    std::ofstream file(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return 0;
    }
    file << std::setprecision(9);
    for (auto d : res) {
        file << d.first << " " << d.second << std::endl;
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
    //return true;
    
    sycl::free(dev_vals, q);
    sycl::free(dev_keys, q);
    sycl::free(hashtable, q);
    sycl::free(hashtable_host, q);
    return 0;
}
