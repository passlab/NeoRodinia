/*
 * This kernel uses a global memory approach, where each thread is responsible for computing a single element in the resulting vector. 
 *
 */
#include "axpy.h"

__global__ void axpy_cudakernel_P3(int N, REAL *Y, const REAL *X, REAL a) {
    __shared__ REAL shared_X[256];
    __shared__ REAL shared_Y[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        shared_X[threadIdx.x] = X[i];  // Load data into shared memory
        shared_Y[threadIdx.x] = Y[i];
        __syncthreads();  // Synchronize threads in a block

        #pragma unroll
        for (int j = 0; j < blockDim.x; j++) {
            shared_Y[j] += a * shared_X[j];  // Perform AXPY in shared memory
        }

        Y[i] = shared_Y[threadIdx.x];  // Store the result back to global memory
    }
}

void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
    REAL *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(REAL));
    cudaMalloc(&d_y, N*sizeof(REAL));

    cudaMemcpy(d_x, X, N*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, Y, N*sizeof(REAL), cudaMemcpyHostToDevice);

    axpy_cudakernel_P3<<<(N+255)/256, 256>>>(d_x, d_y, N, a);
    cudaDeviceSynchronize();

    cudaMemcpy(Y, d_y, N*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

