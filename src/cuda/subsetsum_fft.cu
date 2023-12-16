#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>
#include <cufft.h>
#include "timer.h"

#define BLOCKSIZE 512
#define NAIVE_SIZE 1024

__global__ void subsetSumKernelDp(uint32_t *w, uint32_t *global_dp, const int n, uint32_t sum, uint32_t *blocks) {
    uint32_t T = sum+1;

    uint32_t currentIdx = blockIdx.x * (T);
    uint32_t previousIdx = (1 - blockIdx.x % 2) * T;


    uint32_t interval = (T+BLOCKSIZE - 1)/BLOCKSIZE;
    uint32_t start_index = threadIdx.x*interval;
    uint32_t end_index = start_index+interval+1;
    end_index = (end_index < (T+1)) ? end_index : (T+1);
    

    uint32_t l = NAIVE_SIZE * blockIdx.x;
    uint32_t r = l + NAIVE_SIZE;
    r = (r < n) ? r : (n);
    if (threadIdx.x == 0){
        global_dp[previousIdx]=1;
    }
    __syncthreads();
    for (uint32_t i = l; i < r; i++) {
        const uint32_t x = w[i];

        __syncthreads();

        for (uint32_t dp_block = start_index; dp_block < end_index; dp_block++) {

            if (dp_block >= x) {
                global_dp[currentIdx + dp_block] = global_dp[previousIdx + dp_block] || global_dp[previousIdx + dp_block - x];
            } else {
                global_dp[currentIdx + dp_block] = global_dp[previousIdx + dp_block];
            }
        }

        __syncthreads();

        if (i < (r-1)) {

            uint32_t temp = currentIdx;
            currentIdx = previousIdx;
            previousIdx = temp;
        }

        __syncthreads(); 

        
    }

    for (uint32_t dp_block = start_index; dp_block < end_index; dp_block++) {
        uint32_t flat_idx = blockIdx.x * (T + 1) + threadIdx.x+dp_block;
        blocks[flat_idx] = global_dp[currentIdx + dp_block];
    }
}

__global__ void pointwiseMultiply(cufftDoubleComplex *input1, 
                                  cufftDoubleComplex *input2, 
                                  cufftDoubleComplex *result, 
                                  int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        cufftDoubleComplex a = input1[index];
        cufftDoubleComplex b = input2[index];
        result[index] = make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
}

std::vector<std::complex<double>> fftConvolutionCuFFT(const std::vector<std::complex<double>>& input1,
                         const std::vector<std::complex<double>>& input2) {
    int n = input1.size(); 
    std::vector<std::complex<double>> result(n);
    Timer memory_timer;
    memory_timer.start();

    cufftDoubleComplex *d_input1, *d_input2, *d_result;
    cudaMalloc(&d_input1, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_input2, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_result, n * sizeof(cufftDoubleComplex));


    cudaMemcpy(d_input1, input1.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    memory_timer.end();
    Timer actual_ops;
    actual_ops.start();

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);

    cufftExecZ2Z(plan, d_input1, d_input1, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_input2, d_input2, CUFFT_FORWARD);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    pointwiseMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_result, n);

    cufftExecZ2Z(plan, d_result, d_result, CUFFT_INVERSE);
    actual_ops.end();
    Timer more_mem;
    more_mem.start();

    cudaMemcpy(result.data(), d_result, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_result);
    more_mem.end();
    std::cout << "Time spent on fft ops: " << std::fixed << std::setprecision(10) << (actual_ops.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
    std::cout << "Time spent on memory operations: " << std::fixed << std::setprecision(10) << (more_mem.get_duration<std::chrono::microseconds>() / 1e6 + memory_timer.get_duration<std::chrono::microseconds>() / 1e6 ) << '\n';
    return result;
}

std::vector<std::vector<std::complex<double>>> convertToComplex2D(
    const std::vector<uint32_t>& input, 
    uint32_t num_blocks, 
    uint32_t T) {

    std::vector<std::vector<std::complex<double>>> output(num_blocks, std::vector<std::complex<double>>(T));
    
    for (uint32_t i = 0; i < num_blocks; ++i) {
        for (uint32_t j = 0; j < T; ++j) {
            // Convert input[i * T + j] to std::complex<double>
            double realPart = static_cast<double>(input[i * T + j]);
            output[i][j] = std::complex<double>(realPart, 0.0); // imaginary part is 0
        }
    }

    return output;
}

bool solve_fft(const std::vector<uint32_t>& w, const uint32_t T){

    Timer init_timer;
    init_timer.start();
    const int n = std::size(w);
    uint32_t num_blocks = (n + NAIVE_SIZE - 1) / NAIVE_SIZE;
    //std::cout << "numblocks " << num_blocks << "\n";
    const int num_iterations = std::__lg(num_blocks);
    

    size_t global_dp_size = 2 * (T+1) * num_blocks * sizeof(uint32_t);
    uint32_t* global_dp;
    
    
    // blocks of size num_blocks*sum+1
    std::vector<uint32_t> blocks(num_blocks * (T + 1), 0);
    cudaMalloc(&global_dp, global_dp_size);
    
    uint32_t* d_w;
    cudaMalloc(&d_w, n * sizeof(uint32_t));
    cudaMemcpy(d_w, w.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    

    uint32_t* d_blocks;
    cudaMalloc(&d_blocks, num_blocks*(T+1)*sizeof(uint32_t));

    init_timer.end();
    Timer subsetkernel_timer;
    subsetkernel_timer.start();
    subsetSumKernelDp<<<num_blocks, BLOCKSIZE>>>(d_w, global_dp, n, T, d_blocks);
    subsetkernel_timer.end();
    Timer middleops_timer;
    middleops_timer.start();
    cudaMemcpy(blocks.data(), d_blocks, num_blocks*(T+1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    cudaFree(global_dp);
    cudaFree(d_w);
    
    // create an array of cuda complex doubles
    std::vector fft_outputs(num_iterations + 1, std::vector<std::vector<std::complex<double>>>(num_blocks));
    fft_outputs[0] = convertToComplex2D(blocks, num_blocks, T+1);
    //std::cout << fft_outputs[0][0][5];
    middleops_timer.end();
    Timer fft_merge_timer;
    fft_merge_timer.start();
    for (int iter = 0, iter_num_blocks = num_blocks; iter < num_iterations; iter++, iter_num_blocks = (iter_num_blocks + 1) / 2) {
        const int next_iter_num_blocks = (iter_num_blocks + 1) / 2;

        for (int i = 0; i < next_iter_num_blocks; i++) {
            if (2 * i + 1 < iter_num_blocks) {
                fft_outputs[iter + 1][i] = fftConvolutionCuFFT(fft_outputs[iter][2 * i], fft_outputs[iter][2 * i + 1]);

            } else {
                fft_outputs[iter + 1][i] = std::move(fft_outputs[iter][2 * i]);
            }
        }
    }
    fft_merge_timer.end();

    

    //std::cout << "Time spent combining subarray solutions: " << std::fixed << std::setprecision(10) << (convolution_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';

    bool is_possible = (fft_outputs[num_iterations][0][T].real()>0);

    std::cout << "Time spent on fft_merge: " << std::fixed << std::setprecision(10) << (fft_merge_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
    std::cout << "Time spent on middle_ops: " << std::fixed << std::setprecision(10) << (middleops_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
    std::cout << "Time spent on subsetkernel: " << std::fixed << std::setprecision(10) << (subsetkernel_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
    std::cout << "Time spent on init: " << std::fixed << std::setprecision(10) << (init_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
    return is_possible;

}
