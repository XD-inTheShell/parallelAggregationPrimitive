#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
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
#include <cuda.h>
#include <cuda_runtime.h>
#include "CycleTimer.h"
// #include <cuco/dynamic_map.cuh>
#include "../common.h"
#include "basichash.cuh"
#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <cuda/std/atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuco/static_map.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

// #include "cucohash.cuh"
#define STEP_TSIZE 100 // Set it to maximum size the GPU global memory can hold
#define THREAD_TSIZE 9 // Assuming if no colliding keys, what's the maximum key-value
                        // pair a thread can hold.
#define BLOCKDIMX 16
#define BLOCKDIMY 16
#define BLOCKSIZE BLOCKDIMX*BLOCKDIMY
#define GRIDDIMX 4
#define GRIDDIMY 4
#define GRIDSIZE 16
#define PERTHREADSTEP 1000

template <typename Map>
__global__ void cuCollectAggregateKernel(Map map_view,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread);

void checkCuda(){
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}
int simpleHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap){
    long unsigned int numEntries = keys.size();

    auto constexpr block_size   = BLOCKSIZE;
    auto const grid_size        = GRIDSIZE;
    unsigned int const launch_thread = block_size * grid_size;
    auto const launch_size      = launch_thread * PERTHREADSTEP;

    // Malloc
    Key * device_keys;
    Value * device_values;
    KeyValue* devic_hashtable = create_hashtable();
    KeyValue host_hashtable[KEYSIZE];

    cudaMalloc((void **)&device_keys, sizeof(Key) * launch_size);
    checkCuda();
    cudaMalloc((void **)&device_values, sizeof(Value) * launch_size);
    checkCuda();

    long unsigned int next;

    for (long unsigned int i=0; i<numEntries; i+=launch_size){
        
        unsigned int copySize = std::min((numEntries-i), (long unsigned int)launch_size);
        
        printf("start from %ld, compute size %d\n", i, copySize);
        cudaMemcpy(device_keys, keys.data()+i, sizeof(Key) * copySize, cudaMemcpyHostToDevice);
        checkCuda();
        cudaMemcpy(device_values, values.data()+i, sizeof(Value) * copySize, cudaMemcpyHostToDevice);
        checkCuda();

        double startTime = CycleTimer::currentSeconds();
        simplehashAggregateKernel<<<grid_size, block_size>>>(devic_hashtable,
                device_keys, device_values,
                numEntries, i,  PERTHREADSTEP, launch_thread);
        cudaThreadSynchronize();
        double endTime = CycleTimer::currentSeconds();    
        double overallDuration = endTime - startTime;
        printf("Simple Hash Executed for: %.3f ms\n", 1000.f * overallDuration);
        checkCuda();
    }

    //     unsigned int copySize = 3;
    //     cudaMemcpy(device_keys, keys.data(), sizeof(Key) * copySize, cudaMemcpyHostToDevice);
    //     checkCuda();
    //     cudaMemcpy(device_values, values.data(), sizeof(Value) * copySize, cudaMemcpyHostToDevice);
    //     checkCuda();
    //     printf("grid block %d %d\n", grid_size, block_size);
    // simplehashAggregate<<<grid_size, block_size>>>(devic_hashtable,
    //             device_keys, device_values,
    //             3, 0,  PERTHREADSTEP, launch_thread);
            
    // checkCuda();

    export_hashtable(devic_hashtable, host_hashtable, umap);

    return 0;
}

template <typename Map>
__global__ void cucohashAggregateKernel(Map map_view,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread){

    long unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    long unsigned int start = index * launch_thread;
    long unsigned int offset = index + base;
    for(unsigned int i=index; i<index+step*launch_thread; i+=launch_thread){
        if((i+base) < cap){
            Key key     = device_keys[i];
            Value value = device_values[i];
            // auto slot = map_view.find(device_keys[i]);
            // if(slot!=map_view.end()){
            //     slot->second.fetch_add(value, cuda::memory_order_relaxed);
            // } else{
            //     map_view.insert(key, value);
            // }


            auto [slot, is_new_key] = map_view.insert_and_find({key, value});
            if (!is_new_key) {
                 // key is already in the map -> increment count
                slot->second.fetch_add(value, cuda::memory_order_relaxed);
            }
        }
    }
}

int cucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap){
    long unsigned int numEntries = keys.size();

    auto constexpr block_size   = BLOCKSIZE;
    auto const grid_size        = GRIDSIZE;
    unsigned int const launch_thread = block_size * grid_size;
    auto const launch_size      = launch_thread * PERTHREADSTEP;

    // Malloc
    Key * device_keys;
    // thrust::device_vector<Key> device_keys(numEntries);
    // thrust::copy(keys.begin(), keys.end(), device_keys.begin());
    Value * device_values;
    // thrust::device_vector<Value> device_values(numEntries);
    // thrust::copy(values.begin(), values.end(), device_values.begin());

    cudaMalloc((void **)&device_keys, sizeof(Key) * launch_size);
    checkCuda();
    cudaMalloc((void **)&device_values, sizeof(Value) * launch_size);
    checkCuda();

    std::size_t const capacity = KEYSIZE;
    using Count = Value;
    Key constexpr empty_key_sentinel     = static_cast<Key>(kEmpty);
    Count constexpr empty_value_sentinel = static_cast<Count>(vEmpty);
    cuco::static_map<Key, Count> map{
        capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};
    auto device_insert_view = map.get_device_mutable_view();

    for (long unsigned int i=0; i<numEntries; i+=launch_size){
        
        unsigned int copySize = std::min((numEntries-i), (long unsigned int)launch_size);
        
        printf("start from %ld, compute size %d\n", i, copySize);
        cudaMemcpy(device_keys, keys.data()+i, sizeof(Key) * copySize, cudaMemcpyHostToDevice);
        checkCuda();
        cudaMemcpy(device_values, values.data()+i, sizeof(Value) * copySize, cudaMemcpyHostToDevice);
        checkCuda();
        double startTime = CycleTimer::currentSeconds();
        cucohashAggregateKernel<<<grid_size, block_size>>>(device_insert_view,
                device_keys, device_values,
                numEntries, i,  PERTHREADSTEP, launch_thread);
        cudaThreadSynchronize();
        double endTime = CycleTimer::currentSeconds();    
        double overallDuration = endTime - startTime;
        printf("Cuco Hash Executed for: %.3f ms\n", 1000.f * overallDuration);

        checkCuda();
    }
    // unsigned int copySize = numEntries;
    // cudaMemcpy(device_keys, keys.data(), sizeof(Key) * copySize, cudaMemcpyHostToDevice);
    // checkCuda();
    // cudaMemcpy(device_values, values.data(), sizeof(Value) * copySize, cudaMemcpyHostToDevice);
    // checkCuda();
    // cucohashAggregateKernel<<<grid_size, block_size>>>(device_insert_view,
    //             device_keys, device_values,
    //             numEntries, 0,  PERTHREADSTEP, launch_thread);

    thrust::device_vector<Key> contained_keys(KEYSIZE);
    thrust::device_vector<Value> contained_values(KEYSIZE);
    auto [keyenditr, valenditr] = map.retrieve_all(contained_keys.begin(), contained_values.begin());
    int num = std::distance(contained_keys.begin(), keyenditr);
    for(int i=0; i<num; i++){
        
        Key key = contained_keys[i];
        Value val = contained_values[i];
        // printf("key %u value %u\n", key, val);
        umap[key] = val;
    }
    return 0;
}

// template <typename Map, typename KeyIter, typename ValueIter, typename Predicate>
// __global__ void filtered_insert(Map map_view,
//                                 KeyIter key_begin,
//                                 ValueIter value_begin,
//                                 std::size_t num_keys,
//                                 Predicate pred,
//                                 int* num_inserted)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;

//   std::size_t counter = 0;
//   while (tid < num_keys) {
//     // Only insert keys that pass the predicate
//     // if (pred(key_begin[tid])) {
//     if (1) {
//       // device_mutable_view::insert returns `true` if it is the first time the given key was
//       // inserted and `false` if the key already existed
//       if (map_view.insert({key_begin[tid], value_begin[tid]})) {
//         ++counter;  // Count number of successfully inserted keys
//       }
//     }
//     tid += gridDim.x * blockDim.x;
//   }

//   // Update global count of inserted keys
//   atomicAdd(num_inserted, counter);
// }

// /**
//  * @brief For keys that have a match in the map, increments their corresponding value by one.
//  *
//  * @tparam Map Type of the map returned from static_map::get_device_view
//  * @tparam KeyIter Input iterator whose value_type convertible to Map::key_type
//  *
//  * @param map_view View of the map into which queries will be performed
//  * @param key_begin The beginning of the range of keys to query
//  * @param num_keys The total number of keys
//  */
// template <typename Map>
// __global__ void increment_values(Map map_view, Key * key_begin, std::size_t num_keys)
// {
//   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//   while (tid < num_keys) {
//     // If the key exists in the map, find returns an iterator to the specified key. Otherwise it
//     // returns map.end()
//     auto found = map_view.find(key_begin[tid]);
//     if (found != map_view.end()) {
//       // If the key exists, atomically increment the associated value
//       // The value type of the iterator is pair<cuda::atomic<Key>, cuda::atomic<Value>>
//       found->second.fetch_add(1, cuda::memory_order_relaxed);
//     } else{
//         map_view.insert({key_begin[tid], 3});
//     }
//     tid += gridDim.x * blockDim.x;
//   }
// }

// int test(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap)
// {

//     // // Empty slots are represented by reserved "sentinel" values. These values should be selected such
//     // // that they never occur in your input data.
//     // Key constexpr empty_key_sentinel     = -1;
//     // Value constexpr empty_value_sentinel = -1;

//     // // Number of key/value pairs to be inserted
//     // std::size_t constexpr num_keys = 50'000;

//     // // Create a sequence of keys and values {{0,0}, {1,1}, ... {i,i}}
//     // thrust::device_vector<Key> insert_keys(num_keys);
//     // thrust::sequence(insert_keys.begin(), insert_keys.end(), 0);
//     // thrust::device_vector<Value> insert_values(num_keys);
//     // thrust::sequence(insert_values.begin(), insert_values.end(), 0);

//     // // Compute capacity based on a 50% load factor
//     // auto constexpr load_factor = 0.5;
//     // std::size_t const capacity = std::ceil(num_keys / load_factor);

//     // // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
//     // cuco::static_map<Key, Value> map{
//     // capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

//     // // Get a non-owning, mutable view of the map that allows inserts to pass by value into the kernel
//     // auto device_insert_view = map.get_device_mutable_view();

//     // // Predicate will only insert even keys
//     // auto is_even = [] __device__(auto key) { return (key % 2) == 0; };

//     // // Allocate storage for count of number of inserted keys
//     // thrust::device_vector<int> num_inserted(1);

//     // auto constexpr block_size = 256;
//     // auto const grid_size      = (num_keys + block_size - 1) / block_size;
//     // // filtered_insert<<<grid_size, block_size>>>(device_insert_view,
//     // //                                             insert_keys.begin(),
//     // //                                             insert_values.begin(),
//     // //                                             num_keys,
//     // //                                             is_even,
//     // //                                             num_inserted.data().get());

//     // std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;

//     // // Get a non-owning view of the map that allows find operations to pass by value into the kernel
//     // auto device_find_view = map.get_device_view();
//     // Key * device_keys;
//     // Value * device_values;
//     // cudaMalloc((void **)&device_keys, sizeof(Key) * num_keys);

//     // increment_values<<<grid_size, block_size>>>(device_insert_view, device_keys, num_keys);

//     // // Retrieve contents of all the non-empty slots in the map
//     // thrust::device_vector<Key> contained_keys(num_inserted[0]);
//     // thrust::device_vector<Value> contained_values(num_inserted[0]);
//     // map.retrieve_all(contained_keys.begin(), contained_values.begin());

//     // auto tuple_iter =
//     // thrust::make_zip_iterator(thrust::make_tuple(contained_keys.begin(), contained_values.begin()));
//     // // Iterate over all slot contents and verify that `slot.key + 1 == slot.value` is always true.
//     // auto result = thrust::all_of(
//     // thrust::device, tuple_iter, tuple_iter + num_inserted[0], [] __device__(auto const& tuple) {
//     //     return thrust::get<0>(tuple) + 1 == thrust::get<1>(tuple);
//     // });

//     // if (result) { std::cout << "Success! Target values are properly incremented.\n"; }

//     return 0;
// }