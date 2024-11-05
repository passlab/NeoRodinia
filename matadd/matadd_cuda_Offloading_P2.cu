/*
 * Level 2: 2D Grid of Threads (P2)
 * Description: Uses a 2D grid of threads, where each thread computes one element of the matrix. This approach is more efficient for matrices as it aligns with the 2D nature of the data.
 */
#include "matadd.h"
#include <cuda_runtime.h>

__global__ void matadd_kernel_2D(int N, REAL *C, REAL *A) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] += A[idx];
    }
}

void matadd_kernel(int N, REAL *C, REAL *A) {
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    matadd_kernel_2D<<<grid_size, block_size>>>(N, C, A);
    cudaDeviceSynchronize();
}
