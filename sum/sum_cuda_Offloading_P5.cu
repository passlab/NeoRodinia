/*
 * Have bank conflict in shared memory
 *
 */
#include "sum.h"

/* Have bank conflict in shared memory */
__global__ void sum_with_bank_conflict(const REAL *x, REAL *result) {
    __shared__ REAL cache[ThreadsPerBlock];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = x[tid];
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * cacheIndex;
        if (index < blockDim.x) {
            cache[index] += cache[index + i];
        }
    __syncthreads();
    }
    if (cacheIndex == 0)
    result[blockIdx.x] = cache[cacheIndex];
}

/* second level of reduction */
__global__ void global_final_reduce(REAL* data, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = n/2;
    if (n%2 == 1) {
        stride += 1;
    }
    if (i + stride < n) {
        data[i] += data[i + stride];
    }
}

void final_reduce(REAL* data_device, int n) {
    int residue;
    if (n < (((n+BLOCK_SIZE-1)/BLOCK_SIZE)*((n+BLOCK_SIZE-1)/BLOCK_SIZE))) {
        residue = n;
    } else {
        residue = ((n+BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE;
    }
    while (residue > 1) {
        global_final_reduce<<<(residue+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(data_device, residue);
        if (residue%2 == 1) {
            residue = residue/2 + 1;
        } else {
            residue /= 2;
        }
    }
}

REAL sum_kernel(int N, REAL* A ) {
    REAL *data_device, *output_device, result = 0.0;
    cudaMalloc(&data_device, N*sizeof(REAL));
    cudaMalloc(&output_device, N*sizeof(REAL));

    cudaMemcpy(data_device, A, N*sizeof(REAL), cudaMemcpyHostToDevice);
    
    int BLOCK_NUM = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    sum_with_bank_conflict<<<BLOCK_NUM, BLOCK_SIZE>>>(data_device, output_device);
    final_reduce(output_device, BLOCK_NUM);
    cudaMemcpy(&result, output_device, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(data_device);
    cudaFree(output_device);
    return result;
}
