#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include <map>
struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

const uint32_t kHashTableCapacity = KEYSIZE;
const uint32_t kEmpty = 0xffffffff;
const uint32_t vEmpty = 0x00000000;

__device__ uint32_t hash(uint32_t k);
__device__ void atomicAddValue(KeyValue* hashtable, uint32_t slot, Value value);
__device__ __inline__ void hashtable_update(KeyValue* hashtable, Key key, Value value);
__global__ void hashtable_empty(KeyValue* hashtable);
void print_hashtable(KeyValue* device_hashtable, KeyValue* host_hashtable);
void export_hashtable(KeyValue* device_hashtable, KeyValue* host_hashtable, std::unordered_map<Key, Value> &umap);
KeyValue* create_hashtable();
__global__ void simplehashAggregateKernel(KeyValue* hashtable, 
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread);