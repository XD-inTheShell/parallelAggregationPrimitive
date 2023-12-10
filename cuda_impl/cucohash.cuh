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
#include <cuco/static_map.cuh>
#include "../common.h"

// template <typename Map>
__global__ void cucohashAggregateKernel(cuco::static_map<Key, Value> map_view,
                            Key * device_keys, Value * device_values,
                            long unsigned int cap, long unsigned int base, 
                            unsigned int step, unsigned int const launch_thread);