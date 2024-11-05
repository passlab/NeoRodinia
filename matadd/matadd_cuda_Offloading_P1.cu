/*
 * Level 1: Basic GPU Kernel (P1)
 * Description: A simple 1D grid of threads where each thread computes a single element of the result matrix.
 */
#include "matadd.h"
#include <cuda_runtime.h>

__global__ void matadd_kernel_basic(int N, REAL *C, REAL *A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * N;

    if (idx < total_elements) {
        C[idx] += A[idx];
    }
}

void matadd_kernel(int N, REAL *C, REAL *A) {
    int total_elements = N * N;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    matadd_kernel_basic<<<grid_size, block_size>>>(N, C, A);
    cudaDeviceSynchronize();
}