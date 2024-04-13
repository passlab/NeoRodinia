/*
 * The vector is put in the shared memory. It should be cut to the tiles since the limitation of the capacity of the shared memory.
 *
 */

#include "matvec.h"
#define TILE_SIZE 1024

__global__ void matvec_P3(REAL *matrix, REAL *vector, REAL *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    REAL sum = 0.0;

    __shared__ REAL vector_shared[TILE_SIZE];

    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        // Load one tile of the vector into shared memory
        int idx = tile + threadIdx.x;
        if (idx < n)
            vector_shared[threadIdx.x] = vector[idx];
        else
            vector_shared[threadIdx.x] = 0.0;  // Padding for out-of-bounds indices
        __syncthreads();

        // Compute the partial sum for the current tile
        for (int i = 0; i < TILE_SIZE; i++) {
        int matrix_idx = tid * n + tile + i;
        sum += matrix[matrix_idx] * vector_shared[i];
        }
        __syncthreads();
    }
    result[tid] = sum;
}

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) { REAL *d_matrix, *d_vector, *d_result;
    cudaMalloc(&d_matrix, N * N * sizeof(REAL));
    cudaMalloc(&d_vector, N * sizeof(REAL));
    cudaMalloc(&d_result, N * sizeof(REAL));

    cudaMemcpy(d_matrix, A, N * N * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, B, N * sizeof(REAL), cudaMemcpyHostToDevice);

    // Perform matvec elements
    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;

    matvec_P3<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, N);

    cudaMemcpy(C, d_result, N * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}
