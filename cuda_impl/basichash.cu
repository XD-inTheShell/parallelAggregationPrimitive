// Hash function adapts and modifies based on 
// https://github.com/nosferalatu/SimpleGPUHashTable/tree/master
// which is lock free insert, but do not ensure the modification of
// value thread safe.

#include "basichash.h"
#include "../common.h"
#include <stdlib.h>
#include <stdio.h>
__device__ uint32_t hash(uint32_t k)
{
    // k ^= k >> 16;
    // k *= 0x85ebca6b;
    // k ^= k >> 13;
    // k *= 0xc2b2ae35;
    // k ^= k >> 16;
    // return k & (kHashTableCapacity-1);
}

__device__ void atomicAddValue(KeyValue* hashtable, uint32_t slot, Value value){
    hashtable[slot].value = 1111;
    // Value prevv = hashtable[slot].value;
    // Value writev = prevv + value;
    // while(atomicCAS(&hashtable[slot].value, prevv, writev)!=prevv){
    //     prevv = hashtable[slot].value;
    //     writev = prevv + value;
    // }
    return;
}

// todo: handle VALUE = EMPTYVALUESENTINEL
__device__ void hashtable_update(KeyValue* hashtable, Key key, Value value)
{
    // uint32_t slot = hash(key);

    // while (true)
    // {
    //     Key prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
    //     if (prev == kEmpty)
    //     {
    //         Value prevv = atomicCAS(&hashtable[slot].value, vEmpty, value);
    //         // No thread gets before me
    //         if(prevv == vEmpty){
    //             return;
    //         }
    //         // Some thread with the same key gets before me and wrote its value
    //         // I need to add my value to its value
    //         atomicAddValue(hashtable, slot, value);
    //         // Function only returns if it succesfully added my value, safe to return.
    //         return; 
    //     } else if(prev == key) {
    //         // Some other thread with the same key inserted,
    //         // since we share the same key, I need to atomically add mine.
    //         atomicAddValue(hashtable, slot, value);
    //         return;
    //     }

    //     slot = (slot + 1) & (kHashTableCapacity-1);
    // }
    
}

__global__ void hashtable_empty(KeyValue* hashtable){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < kHashTableCapacity){
        hashtable[index].key    = kEmpty;
        hashtable[index].value  = vEmpty;
    }

    return;
}

void print_hashtable(KeyValue* device_hashtable, KeyValue* host_hashtable){
    cudaMemcpy(host_hashtable, device_hashtable, sizeof(KeyValue) * kHashTableCapacity,
               cudaMemcpyDeviceToHost);
    for(int i=0; i<kHashTableCapacity; i++){
        printf("entry %d: \tkey-%x, value-%x\n", i, host_hashtable[i].key, host_hashtable[i].value);
    }
}

// Host create hashtable
KeyValue* create_hashtable() 
{
    // Allocate memory
    KeyValue* hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);

    // Initialize hash table to empty
    // Since we have a specific pattern we want to set, use a kernel to set it.
    const int threadsPerBlock = 512;
    const int blocks = (kHashTableCapacity + threadsPerBlock - 1) / threadsPerBlock;
    hashtable_empty<<<blocks, threadsPerBlock>>>(hashtable);
    // static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    // cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}
// __device__ bool void hashtable_lookup(KeyValue* hashtable, KeyValue* kvs, unsigned int numkvs)
// {
//     unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (threadid < numkvs)
//     {
//         uint32_t key = kvs[threadid].key;
//         uint32_t slot = hash(key);

//         while (true)
//         {
//             if (hashtable[slot].key == key)
//             {
//                 kvs[threadid].value = hashtable[slot].value;
//                 return;
//             }
//             if (hashtable[slot].key == kEmpty)
//             {
//                 kvs[threadid].value = kEmpty;
//                 return;
//             }
//             slot = (slot + 1) & (kHashTableCapacity - 1);
//         }
//     }
// }