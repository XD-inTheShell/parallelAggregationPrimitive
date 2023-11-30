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
#include <cuco/dynamic_map.cuh>
#include "../common.h"
#define STEP_TSIZE 100 // Set it to maximum size the GPU global memory can hold
#define THREAD_TSIZE 9 // Assuming if no colliding keys, what's the maximum key-value
                        // pair a thread can hold.
#define BLOCKDIMX 32
#define BLOCKDIMY 32
#define BLOCKSIZE BLOCKDIMX*BLOCKDIMY
#define GRIDDIMX 4
#define GRIDDIMY 4
#define GRIDSIZE 16

// __global__ aggregateKernel(int * device_cutoff_info,
//                             int * device_keys, int * device_keys_res,  int * device_res_size
//                             double * device_values, double * device_values_res){
//     int index_inblock = threadIdx.y * blockDim.x + threadIdx.x;
//     int index = (blockIdx.y * gridDim.x + blockIdx.x) * BLOCKSIZE + index_inblock;
//     int task_size = THREAD_TSIZE;
//     if(device_cutoff_info[0]!=-1){
//         if(index > device_cutoff_info[0]){
//             task_size = 0;
//         } else if(index == device_cutoff_info[0]){
//             task_size = device_cutoff_info[1];
//         }
//     }
//     long unsigned int offset = index * THREAD_TSIZE;
//     int* key_offset = device_keys + offset;
//     double* value_offset = device_values + offset;

//     for(int i=0; i<task_size; i++){
        
//     }


    
// }

// NOTE: CAN NOT INSERT 0 VALUE!
int cudaAggregate(std::vector<int> &keys, std::vector<Value> &values, std::unordered_map<int, Value> &umap){
    using Key = int;
    using Count = Value;
    auto constexpr num_keys = KEYSIZE;
    auto constexpr load_factor = 0.5;
    std::size_t const capacity = std::ceil(num_keys / load_factor);
    Key constexpr empty_key_sentinel     = static_cast<Key>(-1);
    Count constexpr empty_value_sentinel = static_cast<Count>(0);
    cuco::static_map<Key, Count> map{
        capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

    long unsigned int numEntries = values.size();
    int perBlockKeyMax = BLOCKSIZE * THREAD_TSIZE;
    int perGridKeyMax = GRIDSIZE * perBlockKeyMax;
    int stepSize = perGridKeyMax;
    printf("stepSize %d\n", stepSize);
    // long unsigned int numEntries = 1;
    int * device_cutoff_info;
    int * device_keys;
    int * device_keys_res;
    int * device_res_size;
    Value * device_values, * device_values_res;
    std::cout << typeid( device_res_size).name() << std::endl;
    std::cout << typeid( device_keys).name() << std::endl;
    int keys_res[stepSize];
    Value values_res[stepSize];
    // Thread ID < cutoff_info[0] should complete THREAD_TSIZE computations
    // Thread ID = cutoff_info[0] should complete cutoff_info[1] compute
    // Thread ID > cutoff_info[0] should not compute 
    int cutoff_info[2];

    cudaMalloc((void **)&device_cutoff_info, sizeof(int)*2);
    cudaMalloc((void **)&device_res_size, sizeof(int));
    cudaMalloc((void **)&device_keys, sizeof(int) * stepSize);
    cudaMalloc((void **)&device_values, sizeof(Value) * stepSize);

    cudaMalloc((void **)&device_keys_res, sizeof(int) * stepSize);
    cudaMalloc((void **)&device_values_res, sizeof(Value) * stepSize);

    for(long unsigned int i=0; i<numEntries;){
        printf("%d\n", numEntries);
        long unsigned int end;
        long unsigned int next;
        long unsigned int count;
        int res_size;
        next = i+stepSize;
        if(next > numEntries){
            count = numEntries - i;
            cutoff_info[0] = count / THREAD_TSIZE;
            cutoff_info[1] = count % THREAD_TSIZE;
        }
        else {
            count = stepSize;
            cutoff_info[0] = -1;
        }

        cudaMemcpy(device_cutoff_info, cutoff_info, sizeof(int) * 2, cudaMemcpyHostToDevice);

        printf("cutoff[0]=%d, cutoff[1]= %d\n", cutoff_info[0], cutoff_info[1]);
        int* key_start = i + keys.data();
        cudaMemcpy(device_keys, key_start, sizeof(int) * count, cudaMemcpyHostToDevice);

        Value* value_start = i + values.data();
        cudaMemcpy(device_values, value_start, sizeof(Value) * count, cudaMemcpyHostToDevice);

        // Call function

        cudaMemcpy(&res_size, device_res_size, sizeof(int), cudaMemcpyDeviceToHost);
        res_size = count;
        // cudaMemcpy(keys_res, device_keys_res, sizeof(int)*res_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(values_res, device_values_res, sizeof(double)*res_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(keys_res, device_keys, sizeof(int)*res_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(values_res, device_values, sizeof(Value)*res_size, cudaMemcpyDeviceToHost);
        // printf("%ld\n", i);
        printf("%ld\n", count);
        for(int j=0; j<res_size; j++){
            int key = keys_res[j];
            Value value = values_res[j];
            auto search = umap.find(key);
            if ( search != umap.end())
                search->second += value;
            else
                umap[key] = value;
        }


        i = next;
    }
    printf("cuda file hi\n");
    return 0;
}