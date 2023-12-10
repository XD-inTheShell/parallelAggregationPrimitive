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
#include "cuda_impl/CycleTimer.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#define KEYSIZE 128

#define VALUEINT
#ifdef VALUEINT
    using Value = uint32_t;
#else
    using Value = double;
#endif
using Key = uint32_t;

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

void checkCuda();