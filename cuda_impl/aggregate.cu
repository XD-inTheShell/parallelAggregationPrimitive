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
// #include <cuco/dynamic_map.cuh>
#include "../common.h"
#include "basichash.h"
#define STEP_TSIZE 100 // Set it to maximum size the GPU global memory can hold
#define THREAD_TSIZE 9 // Assuming if no colliding keys, what's the maximum key-value
                        // pair a thread can hold.
#define BLOCKDIMX 32
#define BLOCKDIMY 32
#define BLOCKSIZE BLOCKDIMX*BLOCKDIMY
#define GRIDDIMX 4
#define GRIDDIMY 4
#define GRIDSIZE 16
// perhaps implement one without interleaving
__global__ void simplehashAggregate(KeyValue* hashtable, 
                            Key * device_keys, Value * device_values,
                            unsigned int cap, unsigned int base, unsigned int step){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int offset = index + base;
    for(unsigned int i=offset; i<offset+step; i++){
        if(index < cap){
            Key key     = device_keys[i];
            Value value = device_values[i];
            hashtable_update(hashtable, key, value);
        }
    }
    

}

void checkCuda(){
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}
int cudaAggregate(std::vector<int> &keys, std::vector<Value> &values, std::unordered_map<int, Value> &umap){
    auto constexpr block_size = BLOCKSIZE;
    auto const grid_size      = GRIDSIZE;
    long unsigned int totalsize = keys.size();
    unsigned int roundstep = (totalsize+(block_size*grid_size))/(block_size*grid_size);

    KeyValue* devic_hashtable = create_hashtable();
    KeyValue host_hashtable[KEYSIZE];

    Key * device_keys;
    Value * device_values;
    // todo: change this
    int copySize = totalsize;
    cudaMalloc((void **)&device_keys, sizeof(unsigned int) * copySize);
    checkCuda();
    cudaMalloc((void **)&device_values, sizeof(Value) * copySize);
    checkCuda();
    cudaMemcpy(device_keys, keys.data(), sizeof(unsigned int) * copySize, cudaMemcpyHostToDevice);
    checkCuda();
    cudaMemcpy(device_values, values.data(), sizeof(Value) * copySize, cudaMemcpyHostToDevice);
    checkCuda();

    simplehashAggregate<<<grid_size, block_size>>>(devic_hashtable,
                device_keys, device_values,
                keys.size(), 0,  roundstep);
    checkCuda();
    // print_hashtable(devic_hashtable, host_hashtable);

    // for(long unsigned int i=0; i<numEntries;){
    //     printf("%d\n", numEntries);
    //     long unsigned int end;
    //     long unsigned int next;
    //     long unsigned int count;
    //     int res_size;
    //     next = i+stepSize;
    //     if(next > numEntries){
    //         count = numEntries - i;
    //         cutoff_info[0] = count / THREAD_TSIZE;
    //         cutoff_info[1] = count % THREAD_TSIZE;
    //     }
    //     else {
    //         count = stepSize;
    //         cutoff_info[0] = -1;
    //     }

    //     cudaMemcpy(device_cutoff_info, cutoff_info, sizeof(int) * 2, cudaMemcpyHostToDevice);

    //     printf("cutoff[0]=%d, cutoff[1]= %d\n", cutoff_info[0], cutoff_info[1]);
    //     int* key_start = i + keys.data();
    //     cudaMemcpy(device_keys, key_start, sizeof(int) * count, cudaMemcpyHostToDevice);

    //     Value* value_start = i + values.data();
    //     cudaMemcpy(device_values, value_start, sizeof(Value) * count, cudaMemcpyHostToDevice);

    //     // Call function
    //     cuCollectAggregateKernel<<<grid_size, block_size>>>(
    //                 device_insert_view, 
    //                 device_cutoff_info,
    //                 device_keys,
    //                 device_values);

    //     i = next;
    // }
    return 0;
}