/*
 * Level 2: SIMD with Memory Alignment (P2)
 * This version includes memory alignment to optimize memory access patterns, ensuring data is aligned properly for SIMD operations.
 * Description: Adds aligned(knodes, keys, records: 32) to optimize memory access and ensure efficient SIMD execution by aligning memory for better performance.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel for
    for (int bid = 0; bid < count; bid++) {
        for (int i = 0; i < maxheight; i++) {
            #pragma omp simd aligned(knodes, start, end: 32)
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
        #pragma omp simd aligned(knodes: 32)
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
        }
        #pragma omp simd aligned(knodes: 32)
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
    }
}
