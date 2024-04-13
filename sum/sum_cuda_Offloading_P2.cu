/*
 * Block distribution of loop iteration
 * Shared memory
 *
 */
#include "sum.h"

/* block distribution of loop iteration, use shared memory */
__global__ void shared_block(REAL* data, REAL* output, int n) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    int element_per_thread = n / total_threads;
    int residue = n % total_threads, start_index, end_index;
    if (thread_id < residue) {
        element_per_thread += 1;
        start_index = element_per_thread * thread_id;
    } else {
        start_index = (element_per_thread + 1) * residue + element_per_thread * (thread_id - residue);
    }

    end_index = start_index + element_per_thread;
    if (end_index > n || (end_index == n && element_per_thread == 0)) {
        end_index = -1;
    }
    int i;
    REAL sum = 0;
    if (end_index != -1) output[thread_id] = 0.0;
    for (i = start_index; i < end_index; i++) {
        sum += data[i];
    }

    int block_thread_id = threadIdx.x;
    __shared__ REAL block_data[BLOCK_SIZE];
    block_data[block_thread_id] = sum;
    __syncthreads();
    for (i = blockDim.x/2; i > 0; i >>= 1) {
        if (block_thread_id < i) {
            sum += block_data[block_thread_id + i];
            block_data[block_thread_id] = sum;
        }
        __syncthreads();
    }

    if (block_thread_id == 0) {
        output[blockIdx.x] = sum;
    }
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
    shared_block<<<BLOCK_NUM, BLOCK_SIZE>>>(data_device, output_device, N);
    final_reduce(output_device, BLOCK_NUM);
    cudaMemcpy(&result, output_device, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(data_device);
    cudaFree(output_device);
    return result;
}
