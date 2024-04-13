/*
 * This kernel uses a global memory approach, where each thread is responsible for computing a single element in the resulting matrix. 
 * The use of thread and block indices to calculate the row and column indices helps in mapping the threads to the elements of the matrices.
 *
 */
#include "matmul.h"
#define BLOCK_SIZE 16

__global__ void global_element(REAL* A, REAL* B, REAL* C, int n) {

    REAL C_value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < n; k++) {
        C_value += A[row * n + k] * B[n * k + col];
    }

    // Each thread writes one element to C matrix
    C[row * n + col] = C_value;
}

void matmul_kernel(int N, REAL* A, REAL* B, REAL* C) {
    REAL *A_device, *B_device, *C_device;
    cudaMalloc(&A_device, N*N*sizeof(REAL));
    cudaMalloc(&B_device, N*N*sizeof(REAL));
    cudaMalloc(&C_device, N*N*sizeof(REAL));

    cudaMemcpy(A_device, A, N*N*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, N*N*sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    global_element<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, N);

    cudaMemcpy(C, C_device, N*N*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}