#include "cucohash.cuh"

// template <typename Map>
__global__ void cucohashAggregateKernel(cuco::static_map<Key, Value> map_view,
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
            auto slot = map_view.find(device_keys[i]);
            if(slot!=map_view.end()){
                slot->second.fetch_add(value, cuda::memory_order_relaxed);
            } else{
                map_view.insert(key, value);
            }
        }
    }
}