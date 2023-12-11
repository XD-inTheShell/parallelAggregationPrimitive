
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



#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

// #include "cucohash.cuh"
void checkCuda(){
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

template <typename Map>
__global__ void cuCollectAggregateKernel(Map map_view,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread);
template <typename Map>
__global__ void localhashCucoaggregate(Map map_view,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread);

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

        cudaThreadSynchronize();
        double startTime = CycleTimer::currentSeconds();
        simplehashAggregateKernel<<<grid_size, block_size>>>(devic_hashtable,
                device_keys, device_values,
                numEntries, i,  computestep, launch_thread);
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

// Only support small key size that fits on the memory
int localHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap){
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
        
        cudaThreadSynchronize();
        double startTime = CycleTimer::currentSeconds();
        localhashAggregate<<<grid_size, block_size>>>(devic_hashtable,
                device_keys, device_values,
                numEntries, i,  computestep, launch_thread);
        cudaThreadSynchronize();
        double endTime = CycleTimer::currentSeconds();    
        double overallDuration = endTime - startTime;
        printf("Local Hash Executed for: %.3f ms\n", 1000.f * overallDuration);
        checkCuda();
    }

    export_hashtable(devic_hashtable, host_hashtable, umap);

    return 0;
}

int localHashnSharedAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap, int debug){
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

        cudaThreadSynchronize();
        double startTime = CycleTimer::currentSeconds();
        localhashSharedAggregate<<<grid_size, block_size>>>(devic_hashtable,
                device_keys, device_values,
                numEntries, i,  computestep, launch_thread);
        cudaThreadSynchronize();
        double endTime = CycleTimer::currentSeconds();    
        double overallDuration = endTime - startTime;
        printf("Local Hash & Shared Hash Executed for: %.3f ms\n", 1000.f * overallDuration);
        checkCuda();
    }

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

        cudaThreadSynchronize();
        double startTime = CycleTimer::currentSeconds();
        cucohashAggregateKernel<<<grid_size, block_size>>>(device_insert_view,
                device_keys, device_values,
                numEntries, i,  computestep, launch_thread);
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

// int localncucoHashAggregate(std::vector<Key> &keys, std::vector<Value> &values, std::unordered_map<Key, Value> &umap){
//     long unsigned int numEntries = keys.size();

//     auto constexpr block_size   = BLOCKSIZE;
//     auto const grid_size        = GRIDSIZE;
//     unsigned int const launch_thread = block_size * grid_size;
//     auto const launch_size      = launch_thread * PERTHREADSTEP;

//     // Malloc
//     Key * device_keys;
//     // thrust::device_vector<Key> device_keys(numEntries);
//     // thrust::copy(keys.begin(), keys.end(), device_keys.begin());
//     Value * device_values;
//     // thrust::device_vector<Value> device_values(numEntries);
//     // thrust::copy(values.begin(), values.end(), device_values.begin());

//     cudaMalloc((void **)&device_keys, sizeof(Key) * launch_size);
//     checkCuda();
//     cudaMalloc((void **)&device_values, sizeof(Value) * launch_size);
//     checkCuda();

//     std::size_t const capacity = KEYSIZE;
//     using Count = Value;
//     Key constexpr empty_key_sentinel     = static_cast<Key>(kEmpty);
//     Count constexpr empty_value_sentinel = static_cast<Count>(vEmpty);
//     cuco::static_map<Key, Count> map{
//         capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};
//     auto device_insert_view = map.get_device_mutable_view();

//     for (long unsigned int i=0; i<numEntries; i+=launch_size){
        
//         unsigned int copySize = std::min((numEntries-i), (long unsigned int)launch_size);
        
//         printf("start from %ld, compute size %d\n", i, copySize);
//         cudaMemcpy(device_keys, keys.data()+i, sizeof(Key) * copySize, cudaMemcpyHostToDevice);
//         checkCuda();
//         cudaMemcpy(device_values, values.data()+i, sizeof(Value) * copySize, cudaMemcpyHostToDevice);
//         checkCuda();
//         double startTime = CycleTimer::currentSeconds();
//         localhashCucoaggregate<<<grid_size, block_size>>>(device_insert_view,
//                 device_keys, device_values,
//                 numEntries, i,  PERTHREADSTEP, launch_thread);
//         cudaThreadSynchronize();
//         double endTime = CycleTimer::currentSeconds();    
//         double overallDuration = endTime - startTime;
//         printf("Cuco Hash Executed for: %.3f ms\n", 1000.f * overallDuration);

//         checkCuda();
//     }
//     // unsigned int copySize = numEntries;
//     // cudaMemcpy(device_keys, keys.data(), sizeof(Key) * copySize, cudaMemcpyHostToDevice);
//     // checkCuda();
//     // cudaMemcpy(device_values, values.data(), sizeof(Value) * copySize, cudaMemcpyHostToDevice);
//     // checkCuda();
//     // cucohashAggregateKernel<<<grid_size, block_size>>>(device_insert_view,
//     //             device_keys, device_values,
//     //             numEntries, 0,  PERTHREADSTEP, launch_thread);

//     thrust::device_vector<Key> contained_keys(KEYSIZE);
//     thrust::device_vector<Value> contained_values(KEYSIZE);
//     auto [keyenditr, valenditr] = map.retrieve_all(contained_keys.begin(), contained_values.begin());
//     int num = std::distance(contained_keys.begin(), keyenditr);
//     for(int i=0; i<num; i++){
        
//         Key key = contained_keys[i];
//         Value val = contained_values[i];
//         // printf("key %u value %u\n", key, val);
//         umap[key] = val;
//     }
//     return 0;
// }
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

__global__ void emtpy_kernel(){
    return;
};
int test()
{   
    cudaThreadSynchronize();
    cudaThreadSynchronize();
    double startTime = CycleTimer::currentSeconds();
    emtpy_kernel<<<GRIDSIZE, BLOCKSIZE>>>();
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();    
    double overallDuration = endTime - startTime;
    printf("*** Empty Executed for: %.3f ms ***\n", 1000.f * overallDuration);

    cudaThreadSynchronize();
    cudaThreadSynchronize();
     startTime = CycleTimer::currentSeconds();
    emtpy_kernel<<<GRIDSIZE, BLOCKSIZE>>>();
    cudaThreadSynchronize();
     endTime = CycleTimer::currentSeconds();    
     overallDuration = endTime - startTime;
    printf("*** Empty Executed for: %.3f ms ***\n", 1000.f * overallDuration);

    cudaThreadSynchronize();
    cudaThreadSynchronize();
     startTime = CycleTimer::currentSeconds();
    emtpy_kernel<<<GRIDSIZE, BLOCKSIZE>>>();
    cudaThreadSynchronize();
     endTime = CycleTimer::currentSeconds();    
     overallDuration = endTime - startTime;
    printf("*** Empty Executed for: %.3f ms ***\n", 1000.f * overallDuration);

    return 0;
}