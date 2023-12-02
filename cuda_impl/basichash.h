#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"

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
__device__ void hashtable_update(KeyValue* hashtable, Key key, Value value);
__global__ void hashtable_empty(KeyValue* hashtable);
void print_hashtable(KeyValue* device_hashtable, KeyValue* host_hashtable);
KeyValue* create_hashtable();