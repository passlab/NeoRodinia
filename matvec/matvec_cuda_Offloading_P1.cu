/*
 * This kernel provides a straightforward mapping of threads to rows, with each thread handling a separate row. 
 * This approach may be more suitable when the number of rows is small or when the matrix is wider than it is tall (m is larger than n).
 *
 */
#include "matvec.h"

__global__ void matvec_P1(REAL *matrix, REAL *vector, REAL *result, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        REAL temp = 0.0;
        for (int j = 0; j < m; j++)
            temp += matrix[i * m + j] * vector[j];
        result[i] = temp;
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
    matvec_P1<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, N, N);

    cudaMemcpy(C, d_result, N * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}
