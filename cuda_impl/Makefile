EXECUTABLE := cudaAggregate
CU_FILES   := basichash.cu aggregate.cu 
CU_DEPS    :=
CC_FILES   := main.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-12.1/lib64/ -lcudart
# LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
IFLAGS=-I/home/xiying/parallelAggregationPrimitive/cuda_impl/include
# IFLAGS=-I/afs/andrew.cmu.edu/usr24/xiyingd/parallelAggregationPrimitive/cuda_impl/include/cuco
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/basichash.o $(OBJDIR)/aggregate.o

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)



$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) $(IFLAGS) -c -o $@