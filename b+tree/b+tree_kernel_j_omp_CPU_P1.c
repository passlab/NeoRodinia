/*
 * Level 1: Basic Parallel Execution (P1)
 * For both kernels (kernel_k and kernel_j), we can parallelize the outermost loop over bid, which processes multiple queries. This ensures that each query is handled in parallel.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel for
    for (int bid = 0; bid < count; bid++) {
        for (int i = 0; i < maxheight; i++) {
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
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
        }
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
    }
}
