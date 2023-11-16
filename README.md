# Parallel Subset Sum (15418 Project)

## TITLE: 
Parallel Subset Sum

Laura Yao, Devin Qu

## URL: 
https://github.com/gss223/15418-project 

## SUMMARY: 
We are going to implement a parallel algorithm for solving the subset sum decision problem where inputs are nonnegative. We will compare the speedup of a CPU implementation using OpenMP and a GPU implementation using CUDA.


## BACKGROUND: 
The subset sum problem (SSP) is an NP-hard decision problem where, given a multiset of integers S and a target value T, one must determine whether some subset of S has a sum of exactly T. However, in this project, we focus on the constrained version where all inputs are positive, a known NP-complete problem. In 1974, Horowitz and Sahni published a O(n * 2^{n/2}) algorithm using meet in the middle by splitting S into two halves, computing all subset sums of each half, and determining whether one sum from each half could sum to T [1]. A more practical approach is the pseudo-polynomial dynamic programming solution running in time O(nT), where for each 0 <= i <= n and 0 <= s <= T, we compute whether a subset of the first i integers can sum to s. In 1999, Pisinger discovered the O(n * max(w_i)) balsub algorithm for solving the knapsack problem, a generalization of SSP [2].

The parts of the algorithm we will focus on optimizing with parallelism are the initial phase of balsub, which reduces the problem size by removing unnecessary elements, and the dynamic programming phase. Since the integer order doesn’t matter, it is possible to partition the subset and combine the results of each subset. Additionally, for small subsets, it is more efficient to use the exponential algorithm, which has very high arithmetic intensity and will likely be faster than the dynamic programming algorithm.

## THE CHALLENGE: 
One challenge lies in the memory usage. Using dynamic programming requires maintaining large tables to keep track of possible sums, so if we divide the set among workers, each worker will need to have a table for their subset of the input. This memory requirement restricts the parallelism amount, especially when values in the multiset are large. Additionally, typical dynamic programming solutions have very poor temporal locality since they usually involve iterating over the multiset and using that integer to update the entire table before moving on to the next one. Another challenge would be in synchronization and dependencies which can be pretty complex in dynamic programming solutions to combine the necessary tables. We also need to be aware of workload imbalance when creating the dynamic programming solution from the division of tables.

## RESOURCES:
We will use the CPUs and GPUs on the GHC machines throughout the development and testing phase. For benchmarking we will use the Bridges-2 RM machines. We will use Pisinger’s paper on balsub as a reference for our sequential solution due to its excellent practical performance. We will also use Curtis and Sanches’s 2015 paper on SSP on GPUs as a reference for our CUDA implementation [3]. For our parallel CPU code, we will begin with the sequential code and design our own implementation.

## GOALS AND DELIVERABLES:
Plan to Achieve
- Sequential solution
- CUDA solution
- OpenMP solution
- Speedup graphs of the parallel implementations over several different workloads
- Profiling of parallel code and identification of bottlenecks

Hope to Achieve
- Further optimizations of parallel implementations
- Alternative parallel frameworks like OpenMPI

We plan to show a live demo of the different versions of the program and speedup graphs at the poster session.


## PLATFORM CHOICE: 
The platform we will be using is with C++ and OpenMP/CUDA to parallelize in CPUs and GPUs respectively. These make sense because we can directly utilize the CUDA structure to map blocks of the dynamic programming structure to the CUDA structure. With OpenMP structures we can also easily parallelize the initialization process with the reduction given. We plan to use the CPUs and GPUs on the GHC machines to easily test the system and Bridges-2 RM machines because of the large amount of cores available.

## SCHEDULE: 
Week 1 (11/13)

Complete proposal, research parallel implementation, begin work on sequential algorithm

Week 2 (11/20)

Finish sequential implementation, begin work on parallel implementations

First work on the OpenMP implementation then the CUDA implementation

Week 3 (11/27)

Continue work on parallel implementation and finish milestone report

Week 4 (12/4)

Finish parallel implementations, collect performance metrics, create speedup graphs

Week 5 (12/11)

Complete final report and poster


[1] Psinger 1999 https://doi.org/10.1006/jagm.1999.1034
[2] https://stackoverflow.com/questions/9962568/fast-solution-to-subset-sum-algorithm-by-pisinger 
[3] Curtis, V. V., & Sanches, C. A. A. (2015). An efficient solution to the subset-sum problem on GPU. Concurrency and Computation: Practice and Experience, 28(1), 95–113. https://doi.org/10.1002/cpe.3636 
