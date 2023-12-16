#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <iomanip>

__global__ void subsetSumKernelRow(uint32_t* dp_current, uint32_t* dp_previous, const uint32_t* set, int i, uint32_t sum) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= sum) {
        if (j >= set[i - 1]) {
            dp_current[j] = dp_previous[j] || dp_previous[j - set[i - 1]];
        } else {
            dp_current[j] = dp_previous[j];
        }
    }
}

bool subsetSumExists(const std::vector<uint32_t>&w, uint32_t T) {
    int n = w.size();

    std::vector<uint32_t> dp(2 * (T + 1), 0);
    dp[0] = 1;

    uint32_t *d_dp_current;
    uint32_t *d_dp_previous;
    cudaMalloc(&d_dp_current, (T + 1) * sizeof(uint32_t));
    cudaMalloc(&d_dp_previous, (T + 1) * sizeof(uint32_t));
    cudaMemcpy(d_dp_previous, dp.data(), (T + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t* d_w;
    cudaMalloc(&d_w, n * sizeof(uint32_t));
    cudaMemcpy(d_w, w.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);


    dim3 dimBlock(256);
    dim3 dimGrid((T + 256 - 1) / 256);

    for (int i = 1; i <= n; ++i) {
        subsetSumKernelRow<<<dimGrid, dimBlock>>>(d_dp_current, d_dp_previous, d_w, i, T);
        cudaDeviceSynchronize();

        std::swap(d_dp_current, d_dp_previous);
    }
    cudaMemcpy(dp.data(), d_dp_previous, (T + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    bool result = dp[T];

    cudaFree(d_dp_current);
    cudaFree(d_dp_previous);
    cudaFree(d_w);

    return result;
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

