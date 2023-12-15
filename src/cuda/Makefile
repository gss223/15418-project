EXECUTABLE := cudaSubsetSum

CU_FILES   := subsetsum_fft.cu
CU_DEPS    :=
CC_FILES   := cuda_fft_test.cpp

###########################################################

ARCH := $(shell uname -s)

OBJDIR := objs
CXX := g++ -m64
CXXFLAGS := -O3 -Wall
LDFLAGS := -L/usr/local/cuda/lib64/ -lcudart -lcufft
NVCC := nvcc
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -ccbin /usr/bin/gcc

OBJS := $(OBJDIR)/cuda_fft_test.o  $(OBJDIR)/subsetsum_fft.o

.PHONY: dirs clean

all: dirs $(EXECUTABLE)

default: all

dirs:
		@mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

