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

#define kHashTableCapacity 2048//16384
#define N 10 * 1000000
#define KEY_EMPTY 0xFFFFFFFF
//#define KEYNUM 1
//#define FACTOR N/KEYNUM
//#define FILL_LEVEL 0.99
#define SORTED 0


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
    //return 0;
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

void insert_nd(kv* hashtable, uint32_t key, uint32_t value)
{
    uint32_t slot = hash(key);

    while (true)
    {
        auto atm = atomic_ref<uint32_t, 
                    memory_order::relaxed,
                    memory_scope::device,
                    access::address_space::generic_space>(hashtable[slot].key);
        uint32_t expected = KEY_EMPTY;
        uint32_t desired = key;
        //atm.fetch_add(1);
        atm.compare_exchange_strong(expected, desired);
        if (expected == KEY_EMPTY)
        {   
            //os << "insert of" << key << " "<< value<< "\n";
            auto ref = atomic_ref<
                uint32_t, 
                memory_order::relaxed,
                memory_scope::device,
                access::address_space::generic_space>(hashtable[slot].value);
            ref.fetch_add(value);
            break;
        }
        else if(expected == key){
            //os << "update of" << key << " "<< value<< "\n";
            auto ref = atomic_ref<
                uint32_t, 
                memory_order::relaxed,
                memory_scope::device,
                access::address_space::generic_space>(hashtable[slot].value);
            ref.fetch_add(value);
            //os << "updated of" << key << " "<< hashtable[slot].value<< "\n";
            break;
        }
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}

void insert_non_atomic(kv* hashtable, uint32_t key, uint32_t value)
{
    uint32_t slot = hash(key);

    while (true)
    {
        //uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        uint32_t prev = hashtable[slot].key;
        if(prev == KEY_EMPTY){
            hashtable[slot].key = key;
            hashtable[slot].value = value;
            break;
        }
        else if(prev == key){
            //os << "update of" << key << " "<< value<< "\n";
            hashtable[slot].value += value;
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
    auto selector = default_selector_v;
    // auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    // for(auto &device : devices) {
    //     sycl::queue q(device);
    //     auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    //     auto prefer_group_sizes = device.get_info<sycl::info::device::preferred_work_group_size>();
    //     std::cout << prefer_group_sizes << "\n";
    // }
    //auto selector = cpu_selector_v;

    queue q(selector, exception_handler);

    std::cerr << "Running on device: " << q.get_device().get_info<info::device::name>() << std::endl;

    //std::cerr << "hi" << std::endl;
    std::vector<uint32_t> vec_keys;
    std::vector<uint32_t> vec_values;
    //readFile("../testcases/inputs/in_sorted.txt", vec_keys, vec_values, N);
    // uint32_t keys[N];
    // uint32_t values[N];
    //uint32_t *keys = (uint32_t *)std::malloc(sizeof(uint32_t)*N);
    //uint32_t *values = (uint32_t *)std::malloc(sizeof(uint32_t)*N);;
    //std::copy(vec_keys.begin(), vec_keys.end(), keys);
    //std::copy(vec_values.begin(), vec_values.end(), values);
    float FILL_LEVELS[9] = {0.25,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99};
    // int INPUT_SIZES[6] = 
    // {

    //     //5 * 100,
    //     5 * 1000,
    //     5 * 10000,
    //     5 * 100000,
    //     5 * 1000000,
    //     5 * 10000000,
    //     5 * 100000000
    // };
    for(int k = 0; k < 9; k++){
    //int N = INPUT_SIZES[k];
    float FILL_LEVEL = FILL_LEVELS[k];
    uint32_t *dev_keys;
    uint32_t *dev_vals;
    kv *hashtable;
    kv *hashtable_host;

    dev_keys = sycl::malloc_device<uint32_t>(N, q);
    dev_vals = sycl::malloc_device<uint32_t>(N, q);
    hashtable = sycl::malloc_device<kv>(kHashTableCapacity, q);
    hashtable_host = sycl::malloc_host<kv>(kHashTableCapacity, q);

    //buffer<uint32_t> buf_keys(keys, N);
    //buffer<uint32_t> buf_values(values, N);
    int KEYNUM = kHashTableCapacity * FILL_LEVEL;
    std::cout << "\n\n\n"<< FILL_LEVEL << "  "<< KEYNUM << "\n";
    int factor = (N / KEYNUM) + 1;
    if(SORTED){
        q.submit([&](handler& h){
            h.parallel_for(N, [=](item<1> j){
                dev_keys[j] = j / factor;
                dev_vals[j] = 1;
            });
        });
    }
    else{
        q.submit([&](handler& h){
            h.parallel_for(N, [=](item<1> j){
                dev_keys[j] = j%KEYNUM;
                dev_vals[j] = 1;
            });

        });
    }
    
    
    //q.memcpy(hashtable_host, hashtable, sizeof(kv)*kHashTableCapacity).wait();
    //q.memcpy(dev_keys, keys, sizeof(uint32_t)*N).wait();
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    q.submit([&](handler& h){
        h.parallel_for(kHashTableCapacity, [=](item<1> i){
            hashtable[i].key = KEY_EMPTY;
            hashtable[i].value = 0;
        });
    });
    
    q.wait();
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
    //q.memcpy(hashtable_host, hashtable, sizeof(kv)*kHashTableCapacity).wait();
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> submission_time = t1 - t0;
    std::chrono::duration<double> total_time = t2 - t0;
    //std::chrono::duration<double> diff = t2-t1;
    q.memcpy(hashtable_host, hashtable, sizeof(kv)*kHashTableCapacity).wait();
    std::cout << "SIMPLE HASH\n";
    // std::cout << "submission time " << submission_time.count()*1000 << " ms" << "\n";
    // std::cout << "total time " << total_time.count()*1000 << " ms" << "\n";
    std::cout << "throughput " << (N/total_time.count())/1e9 << " GPairs/s\n";
    //std::cout << "diff" << diff.count()*1000 << "ms\n";

    std::map<uint32_t, uint32_t> res;
    for(int i = 0; i < kHashTableCapacity; i++){
        if(hashtable_host[i].key != KEY_EMPTY){
            //std::cerr << hashtable_host[i].key << "  " << hashtable_host[i].value << "\n";
            res.insert({hashtable_host[i].key, hashtable_host[i].value});
        }
    }
    
    std::string fileName = "out_simple.txt";
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

    ///////////////////////////////////////////////////////////////////////////////
    q.submit([&](handler& h){
        h.parallel_for(kHashTableCapacity, [=](item<1> i){
            hashtable[i].key = KEY_EMPTY;
            hashtable[i].value = 0;
        });
    });
    const int BLOCK_X = 64;
    const int BLOCK_Y = 16;
    const int GRID_X = 128;
    const int GRID_Y = 1;
    const int TOTAL_THREADS = BLOCK_X * BLOCK_Y * GRID_X * GRID_Y;
    const int ENTRIES_PER_THREAD = (N / TOTAL_THREADS) + 1;
    q.wait();
    t0 = std::chrono::steady_clock::now();
    q.submit([&](handler& h){
        //stream os(1024, 128, h);
        range<3> grid_dim(BLOCK_X*GRID_X,BLOCK_Y*GRID_Y,1);//get by get_global_range()
        range<3> block_dim(BLOCK_X,BLOCK_Y,1);// get_local_range()
        sycl::local_accessor<kv> hash_acc(sycl::range<1>((kHashTableCapacity)), h);
        h.parallel_for(nd_range<3>(grid_dim, block_dim), [=](nd_item<3> it){
            int my_thread_id = it.get_local_id(1)*BLOCK_X + it.get_local_id(0);
            int my_block_id = it.get_group(1)*GRID_X + it.get_group(0);
            
            
            kv *table = hash_acc.get_pointer();
            uint32_t key;
            uint32_t value;
            for(int i = my_thread_id; i < kHashTableCapacity; i += BLOCK_X * BLOCK_Y){
                table[i].key = KEY_EMPTY;
                table[i].value = 0;
            }
            sycl::group_barrier(it.get_group());
            int idx_start = my_block_id * BLOCK_X * BLOCK_Y;
            for(int i = idx_start; i < N; i += TOTAL_THREADS){
                if(i+my_thread_id >= N){
                    break;
                }
                key = dev_keys[i+my_thread_id];
                value = dev_vals[i+my_thread_id];
                uint32_t slot = lookup(table, key);
                if (slot != KEY_EMPTY){
                    auto ref = atomic_ref<
                        uint32_t, 
                        memory_order::relaxed,
                        memory_scope::device,
                        access::address_space::generic_space>(table[slot].value);
                    ref.fetch_add(value);
                    //table[slot].value += value;
                }
                else{
                    //insert_non_atomic(table, key, value);
                    insert_nd(table, key, value);
                }
                //os << slot <<"\n";
            }
            sycl::group_barrier(it.get_group());
            if(my_thread_id < kHashTableCapacity){
                    if(table[my_thread_id].key != KEY_EMPTY){
                        
                        uint32_t key = table[my_thread_id].key;
                        uint32_t value = table[my_thread_id].value;
                        //insert(hashtable, key, value);
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
                    }
            }
            
        });

    });
    t1 = std::chrono::steady_clock::now();
    q.wait();
    
    t2 = std::chrono::steady_clock::now();
    submission_time = t1 - t0;
    total_time = t2 - t0;
    //diff = t2-t1;
    q.memcpy(hashtable_host, hashtable, sizeof(kv)*kHashTableCapacity).wait();

    std::cout << "shared hashtable (all threads a block share one)\n";
    // std::cout << "submission time " << submission_time.count()*1000 << " ms" << "\n";
    // std::cout << "total time " << total_time.count()*1000 << " ms" << "\n";
    std::cout << "throughput " << (N/total_time.count())/1e9 << " GPairs/s\n";
    //std::cout << "diff" << diff.count()*1000 << "ms\n";

    
    std::map<uint32_t, uint32_t> res1;
    for(int i = 0; i < kHashTableCapacity; i++){
        if(hashtable_host[i].key != KEY_EMPTY){
            //std::cerr << hashtable_host[i].key << "  " << hashtable_host[i].value << "\n";
            res1.insert({hashtable_host[i].key, hashtable_host[i].value});
        }
    }

    fileName = "out.txt";
    file.open(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return 0;
    }
    file << std::setprecision(9);
    for (auto d : res1) {
        file << d.first << " " << d.second << std::endl;
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;

    
    
    sycl::free(dev_vals, q);
    sycl::free(dev_keys, q);
    sycl::free(hashtable, q);
    sycl::free(hashtable_host, q);
    }
    return 0;
}
