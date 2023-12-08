CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -Iinclude

ISPC = ispc
ISPCFLAGS = -O3 --target=avx2-i32x8 --arch=x86-64

ifeq ($(mode),release)
	CXXFLAGS += -O3 -mavx2
	ISPCFLAGS += -O3
else ifeq ($(mode),debug)
	CXXFLAGS += -g -O0 -D_GLIBCXX_DEBUG
endif

SRCDIR = src
OBJDIR = build
BINDIR = bin

.PHONY: all clean object bin

parallel_cpu_test: parallel_cpu.o get_subset_sums.o convolution.o utils.o butterfly.o parallel_cpu_test.o naive.o bin
	$(CXX) $(CXXFLAGS) -fopenmp -o bin/parallel_cpu_test $(OBJDIR)/parallel_cpu.o $(OBJDIR)/get_subset_sums.o $(OBJDIR)/convolution.o $(OBJDIR)/utils.o $(OBJDIR)/butterfly.o $(OBJDIR)/parallel_cpu_test.o $(OBJDIR)/naive.o

naive_test: naive_test.o naive.o bin
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/naive_test $(OBJDIR)/naive_test.o $(OBJDIR)/naive.o

parallel_cpu_test.o: object
	$(CXX) $(CXXFLAGS) -c -o $(OBJDIR)/parallel_cpu_test.o $(SRCDIR)/parallel_cpu_test.cpp

naive_test.o: object
	$(CXX) $(CXXFLAGS) -c -o $(OBJDIR)/naive_test.o $(SRCDIR)/naive_test.cpp

parallel_cpu.o: object
	$(CXX) $(CXXFLAGS) -fopenmp -c -o $(OBJDIR)/parallel_cpu.o $(SRCDIR)/parallel_cpu.cpp

convolution.o: object
	$(CXX) $(CXXFLAGS) -fopenmp -c -o $(OBJDIR)/convolution.o $(SRCDIR)/convolution.cpp

utils.o: object
	$(CXX) $(CXXFLAGS) -fopenmp -c -o $(OBJDIR)/utils.o $(SRCDIR)/utils.cpp

butterfly.o: object
	$(ISPC) $(ISPCFLAGS) -o $(OBJDIR)/butterfly.o $(SRCDIR)/butterfly.ispc

get_subset_sums.o: object
	$(ISPC) $(ISPCFLAGS) -o $(OBJDIR)/get_subset_sums.o $(SRCDIR)/get_subset_sums.ispc

naive.o: object
	$(CXX) $(CXXFLAGS) -c -o $(OBJDIR)/naive.o $(SRCDIR)/naive.cpp

gen: bin
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/gen $(SRCDIR)/gen.cpp

bin:
	@mkdir -p $(BINDIR)

object:
	@mkdir -p $(OBJDIR)

clean:
	rm -f bin/* $(OBJDIR)/*.o logs/*
