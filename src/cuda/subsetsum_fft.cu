#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>
#include <cufft.h>

#define BLOCKSIZE 256
#define NAIVE_SIZE 2048

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

        __syncthreads(); // Synchronize before starting to read/write

        for (uint32_t dp_block = start_index; dp_block < end_index; dp_block++) {
            // Compute dp_current based on dp_previous
            if (dp_block >= x) {
                global_dp[currentIdx + dp_block] = global_dp[previousIdx + dp_block] || global_dp[previousIdx + dp_block - x];
            } else {
                global_dp[currentIdx + dp_block] = global_dp[previousIdx + dp_block];
            }
        }

        __syncthreads(); // Synchronize after writing to global memory

        // Swap logic: switch currentIdx and previousIdx for the next iteration
        if (i < (r-1)) {
            // Swap the indices
            uint32_t temp = currentIdx;
            currentIdx = previousIdx;
            previousIdx = temp;
        }

        __syncthreads(); // Synchronize before starting the next iteration

        
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
    int n = input1.size(); // Assuming input1 and input2 are the same size
    std::vector<std::complex<double>> result(n);

    // Allocate memory on the device
    cufftDoubleComplex *d_input1, *d_input2, *d_result;
    cudaMalloc(&d_input1, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_input2, n * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_result, n * sizeof(cufftDoubleComplex));

    // Copy host data to device
    cudaMemcpy(d_input1, input1.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    // Create a cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);

    // Execute forward FFT
    cufftExecZ2Z(plan, d_input1, d_input1, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_input2, d_input2, CUFFT_FORWARD);

    // Perform point-wise multiplication
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    pointwiseMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_result, n);

    // Execute inverse FFT
    cufftExecZ2Z(plan, d_result, d_result, CUFFT_INVERSE);

    // Copy result back to host
    cudaMemcpy(result.data(), d_result, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    // Clean up
    cufftDestroy(plan);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_result);
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

    
    subsetSumKernelDp<<<num_blocks, BLOCKSIZE>>>(d_w, global_dp, n, T, d_blocks);
    
    cudaMemcpy(blocks.data(), d_blocks, num_blocks*(T+1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    cudaFree(global_dp);
    cudaFree(d_w);
    
    // create an array of cuda complex doubles
    std::vector fft_outputs(num_iterations + 1, std::vector<std::vector<std::complex<double>>>(num_blocks));
    fft_outputs[0] = convertToComplex2D(blocks, num_blocks, T+1);
    //std::cout << fft_outputs[0][0][5];
    
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

    

    //std::cout << "Time spent combining subarray solutions: " << std::fixed << std::setprecision(10) << (convolution_timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';

    bool is_possible = (fft_outputs[num_iterations][0][T].real()>0);

    
    
    return is_possible;

}
