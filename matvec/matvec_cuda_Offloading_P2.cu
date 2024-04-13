/*
 * Introduce a grid-stride loop to distribute the workload across multiple threads and blocks. 
 * This can be more efficient when the number of rows is large, as it allows for better utilization of GPU resources and may lead to better parallelism.
 *
 */

#include "matvec.h"

__global__ void matvec_P2(REAL *matrix, REAL *vector, REAL *result, int n, int m) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    int elementsPerBlock = (n + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elementsPerBlock; i++) {
        int rowIndex = threadId + i * blockDim.x;

        if (rowIndex < n) {
            REAL temp = 0.0;
            for (int j = 0; j < m; j++) {
                temp += matrix[rowIndex * m + j] * vector[j];
            }
            result[rowIndex] = temp;
        }
    }
}

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
  REAL *d_matrix, *d_vector, *d_result;
  cudaMalloc(&d_matrix, N * N * sizeof(REAL));
  cudaMalloc(&d_vector, N * sizeof(REAL));
  cudaMalloc(&d_result, N * sizeof(REAL));

  cudaMemcpy(d_matrix, A, N * N * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, B, N * sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform matvec elements
  int blockSize = 1024;
  int gridSize = (N + blockSize - 1) / blockSize;

  matvec_P2<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, N, N);

  cudaMemcpy(C, d_result, N * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_result);
}
