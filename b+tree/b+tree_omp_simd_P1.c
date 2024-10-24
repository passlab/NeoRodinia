/*
 * Level 1: Basic SIMD Vectorization (P1)
 * For SIMD parallelism, the inner loop of the tree processing can be vectorized using the simd directive to optimize the traversal across the nodes.
 * Description: The SIMD directive vectorizes the inner loops for both kernel_k and kernel_j to maximize loop-level parallelism.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {
    #pragma omp parallel for
    for (int bid = 0; bid < count; bid++) {
        for (int i = 0; i < maxheight; i++) {
            #pragma omp simd
            for (int thid = 0; thid < threadsPerBlock; thid++) {
                if ((knodes[currKnode[bid]].keys[thid] <= keys[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
                    if (knodes[offset[bid]].indices[thid] < knodes_elem) {
                        offset[bid] = knodes[offset[bid]].indices[thid];
                    }
                }
            }
            currKnode[bid] = offset[bid];
        }
        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
            }
        }
    }
}

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel for
    for (int bid = 0; bid < count; bid++) {
        for (int i = 0; i < maxheight; i++) {
            #pragma omp simd
            for (int thid = 0; thid < threadsPerBlock; thid++) {
                if ((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > start[bid])) {
                    if (knodes[currKnode[bid]].indices[thid] < knodes_elem) {
                        offset[bid] = knodes[currKnode[bid]].indices[thid];
                    }
                }
                if ((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid + 1] > end[bid])) {
                    if (knodes[lastKnode[bid]].indices[thid] < knodes_elem) {
                        offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
                    }
                }
            }
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
        }
        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
        }
        #pragma omp simd
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
    }
}
