/* 
 * This code snippet is a mix of C and CUDA C code for implementing operations on a B+ tree data structure using GPU acceleration. 
 * `findK` kernel function: This CUDA kernel function is used to find a specific key in the B+ tree. It iterates through the tree levels and compares the keys to find the target key. If found, it updates the corresponding record.
 * `findRangeK` kernel function: This CUDA kernel function is used to find a range of keys in the B+ tree. It iterates through the tree levels and compares the keys to find the starting and ending indices of the range. It then calculates the length of the range.
 * `kernel_k` function: This function sets up and launches the `findK` kernel to find a specific key in the B+ tree.
 * `kernel_j` function: This function sets up and launches the `findRangeK` kernel to find a range of keys in the B+ tree.
 *
 */
 
#include <cuda.h>
#include "b+tree.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void findK(long height, knode *knodesD, long knodes_elem, record *recordsD, long *currKnodeD, long *offsetD, int *keysD, record *ansD) {

    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // process tree levels
    for (int i = 0; i < height; i++) {

        // if value is between the two keys
        if ((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] && 
            (knodesD[currKnodeD[bid]].keys[thid + 1] > keysD[bid])) {

            // avoid segmentation fault by checking bounds
            if (knodesD[offsetD[bid]].indices[thid] < knodes_elem) {
                offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnodeD[bid] = offsetD[bid];
        }
        __syncthreads();
    }

    // Check each key to find the target record
    if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]) {
        ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
    }
}

__global__ void findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, 
                           long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD) {
    // private thread IDs
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // process tree levels
    for (int i = 0; i < height; i++) {
        if ((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid + 1] > startD[bid])) {
            if (knodesD[currKnodeD[bid]].indices[thid] < knodes_elem) {
                offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
            }
        }
        if ((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid + 1] > endD[bid])) {
            if (knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem) {
                offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
            }
        }
        __syncthreads();

        // set for next tree level
        if (thid == 0) {
            currKnodeD[bid] = offsetD[bid];
            lastKnodeD[bid] = offset_2D[bid];
        }
        __syncthreads();
    }

    // Find the index of the starting record
    if (knodesD[currKnodeD[bid]].keys[thid] == startD[bid]) {
        RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
    }
    __syncthreads();

    // Find the index of the ending record
    if (knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]) {
        ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid] + 1;
    }
}


void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, 
              long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {

    int numBlocks;
    numBlocks = count; // max # of blocks can be 65,535

    // Allocate device memory
    record *recordsD;
    cudaMalloc((void**)&recordsD, records_mem);

    knode *knodesD;
    cudaMalloc((void**)&knodesD, knodes_mem);

    long *currKnodeD;
    cudaMalloc((void**)&currKnodeD, count * sizeof(long));

    long *offsetD;
    cudaMalloc((void**)&offsetD, count * sizeof(long));

    int *keysD;
    cudaMalloc((void**)&keysD, count * sizeof(int));

    record *ansD;
    cudaMalloc((void**)&ansD, count * sizeof(record));

    // Copy data from host to device
    cudaMemcpy(recordsD, records, records_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(currKnodeD, currKnode, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(keysD, keys, count * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    findK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, recordsD, currKnodeD, offsetD, keysD, ansD);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(ans, ansD, count * sizeof(record), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(recordsD);
    cudaFree(knodesD);
    cudaFree(currKnodeD);
    cudaFree(offsetD);
    cudaFree(keysD);
    cudaFree(ansD);
}

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, 
              long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    int numBlocks;
    numBlocks = count;

    knode *knodesD;
    cudaMalloc((void**)&knodesD, knodes_mem);

    long *currKnodeD;
    cudaMalloc((void**)&currKnodeD, count * sizeof(long));

    long *offsetD;
    cudaMalloc((void**)&offsetD, count * sizeof(long));

    long *lastKnodeD;
    cudaMalloc((void**)&lastKnodeD, count * sizeof(long));

    long *offset_2D;
    cudaMalloc((void**)&offset_2D, count * sizeof(long));

    int *startD;
    cudaMalloc((void**)&startD, count * sizeof(int));

    int *endD;
    cudaMalloc((void**)&endD, count * sizeof(int));

    int *ansDStart;
    cudaMalloc((void**)&ansDStart, count * sizeof(int));
    int *ansDLength;
    cudaMalloc((void**)&ansDLength, count * sizeof(int));
    cudaMemcpy(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(currKnodeD, currKnode, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(lastKnodeD, lastKnode, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(offset_2D, offset_2, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(startD, start, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(endD, end, count * sizeof(int), cudaMemcpyHostToDevice);

    findRangeK<<<numBlocks, threadsPerBlock>>>(maxheight, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD, offset_2D, startD, endD, ansDStart, ansDLength);
    cudaDeviceSynchronize();

    cudaMemcpy(recstart, ansDStart, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(reclength, ansDLength, count * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(knodesD);
    cudaFree(currKnodeD);
    cudaFree(offsetD);
    cudaFree(lastKnodeD);
    cudaFree(offset_2D);
    cudaFree(startD);
    cudaFree(endD);
    cudaFree(ansDStart);
    cudaFree(ansDLength);
}

#ifdef __cplusplus
}
#endif
