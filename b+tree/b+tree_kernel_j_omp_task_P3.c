/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Uses recursive tasks for hierarchical traversal through the B+ tree, creating a task for each level in the traversal.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_j_recursive(knode *knodes, long knodes_elem, int order, long maxheight, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int bid, int level, int threadsPerBlock) {
    if (level >= maxheight) {
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
        return;
    }

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

    #pragma omp task
    kernel_j_recursive(knodes, knodes_elem, order, maxheight, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength, bid, level + 1, threadsPerBlock);
}

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task
                kernel_j_recursive(knodes, knodes_elem, order, maxheight, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength, bid, 0, threadsPerBlock);
            }
        }
    }
}
