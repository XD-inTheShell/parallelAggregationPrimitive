// Hash function adapts and modifies based on 
// https://github.com/nosferalatu/SimpleGPUHashTable/tree/master
// which is lock free insert, but do not ensure the modification of
// value thread safe.

#include "basichash.cuh"
#include "../common.h"
#include <stdlib.h>
#include <stdio.h>

// The hashmaps in this implementation assumes that hash key can 
// not be erased.
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}
// Local Hashtable ------------------------------
__device__ void localhashUpdate(KeyValue* privateHashtable,
                                 uint32_t key, uint32_t value){
    uint32_t slot = hash(key);
    while (true)
    {
        // Insert myself
        if (privateHashtable[slot].key==kEmpty){
            privateHashtable[slot].key = key;
            privateHashtable[slot].value = value;
            return;
        } 
        // Insertion failed, check if this is my slot 
        else if(privateHashtable[slot].key==key){
            privateHashtable[slot].value += value;
            return;
        }   
        slot = (slot + 1) & (kHashTableCapacity-1);
    }
}
__global__ void localhashAggregate(KeyValue* globalHashtable,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread){

    KeyValue privateHashtable[KEYSIZE];
    for(int i=0; i<KEYSIZE; i++){
        privateHashtable[i].key = kEmpty;
        // privateHashtable[i].value = vEmpty;
    }

    long unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i=index; i<index+step*launch_thread; i+=launch_thread){
        if((i+base) < cap){
            Key key     = device_keys[i];
            Value value = device_values[i];
            localhashUpdate(privateHashtable, key, value);
        }
    }

    // Write to global hash
    for(int i=0; i<KEYSIZE; i++){
        Key key = privateHashtable[i].key;
        if(key!=kEmpty){
            Value value = privateHashtable[i].value;
            hashtable_update(globalHashtable, key, value);
        }
    }
}
// Global Hashtable -----------------------------
__device__ void atomicAddValue(KeyValue* hashtable, uint32_t slot, Value value){
    // hashtable[slot].value = 1111;
    Value prevv = hashtable[slot].value;
    Value writev = prevv + value;
    while(atomicCAS(&hashtable[slot].value, prevv, writev)!=prevv){
        prevv = hashtable[slot].value;
        writev = prevv + value;
    }
    return;
}

// todo: handle VALUE = EMPTYVALUESENTINEL
__device__  __inline__ void hashtable_update(KeyValue* hashtable, Key key, Value value)
{
    uint32_t slot = hash(key);

    while (true)
    {
        Key prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        if (prev == kEmpty)
        {
            Value prevv = atomicCAS(&hashtable[slot].value, vEmpty, value);
            // No thread gets before me
            if(prevv == vEmpty){
                return;
            }
            // Some thread with the same key gets before me and wrote its value
            // I need to add my value to its value
            atomicAddValue(hashtable, slot, value);
            // Function only returns if it succesfully added my value, safe to return.
            return; 
        } else if(prev == key) {
            // Some other thread with the same key inserted,
            // since we share the same key, I need to atomically add mine.
            atomicAddValue(hashtable, slot, value);
            return;
        }

        slot = (slot + 1) & (kHashTableCapacity-1);
    }
    
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
        printf("entry %d: \tkey=%x, value=%x\n", i, host_hashtable[i].key, host_hashtable[i].value);
    }
}

void export_hashtable(KeyValue* device_hashtable, KeyValue* host_hashtable, std::unordered_map<Key, Value> &umap){
    cudaMemcpy(host_hashtable, device_hashtable, sizeof(KeyValue) * kHashTableCapacity,
               cudaMemcpyDeviceToHost);
    for(int i=0; i<kHashTableCapacity; i++){
        if(host_hashtable[i].key!=kEmpty){
            // printf("entry %d: \tkey=%u, value=%u\n", i, host_hashtable[i].key, host_hashtable[i].value);
            umap[host_hashtable[i].key]=host_hashtable[i].value;
        }
        
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

__global__ void simplehashAggregateKernel(KeyValue* hashtable, 
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread){
    long unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    long unsigned int start = index * launch_thread;
    long unsigned int offset = index + base;
    // Key key     = device_keys[index];
    //         Value value = device_values[index];
    //         hashtable_update(hashtable, key, index);
    for(unsigned int i=index; i<index+step*launch_thread; i+=launch_thread){
        if((i+base) < cap){
            Key key     = device_keys[i];
            Value value = device_values[i];
            hashtable_update(hashtable, key, value);
        }
    }
    

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