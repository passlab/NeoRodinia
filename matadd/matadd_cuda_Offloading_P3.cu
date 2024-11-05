/*
 * Level 3: Tiled Matrix Addition with Shared Memory (P3)
 * Description: Uses shared memory to load tiles of the matrix, improving memory access efficiency by reducing global memory accesses.
 */
#include "matadd.h"
#include <cuda_runtime.h>

__global__ void matadd_kernel_tiled(int N, REAL *C, REAL *A) {
    __shared__ REAL tile_A[16][16];
    __shared__ REAL tile_C[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;

        // Load data into shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[idx];
        tile_C[threadIdx.y][threadIdx.x] = C[idx];
        __syncthreads();

        // Perform matrix addition in shared memory
        tile_C[threadIdx.y][threadIdx.x] += tile_A[threadIdx.y][threadIdx.x];
        __syncthreads();

        // Write result back to global memory
        C[idx] = tile_C[threadIdx.y][threadIdx.x];
    }
}

void matadd_kernel(int N, REAL *C, REAL *A) {
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    matadd_kernel_tiled<<<grid_size, block_size>>>(N, C, A);
    cudaDeviceSynchronize();
}