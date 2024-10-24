/*
 * This kernel uses a global memory approach, where each thread is responsible for computing a single element in the resulting vector. 
 *
 */
#include "axpy.h"

__global__ void axpy_cudakernel_P2(int N, REAL *Y, const REAL *X, REAL a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] += a * X[i];  // Ensures coalesced access
    }
}

void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
    REAL *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(REAL));
    cudaMalloc(&d_y, N*sizeof(REAL));

    cudaMemcpy(d_x, X, N*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, Y, N*sizeof(REAL), cudaMemcpyHostToDevice);

    axpy_cudakernel_P2<<<(N+255)/256, 256>>>(d_x, d_y, N, a);
    cudaDeviceSynchronize();

    cudaMemcpy(Y, d_y, N*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

