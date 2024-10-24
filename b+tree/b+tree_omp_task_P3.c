/*
 * Level 3: Recursive Tasking (P3)
 * In this version, we use recursive tasking to break down the work into smaller subtasks. This can be useful when the workload is highly irregular, allowing better dynamic load balancing.
 * Description: Recursive tasking breaks the traversal into subtasks for each level of the tree, allowing more dynamic load balancing and parallel execution.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_task(int bid, int maxheight, int threadsPerBlock, long knodes_elem, long *currKnode, long *offset, knode *knodes, int *keys, record *records, record *ans) {
    if (maxheight == 0) {  // Base case
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
            }
        }
    } else {
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if ((knodes[currKnode[bid]].keys[thid] <= keys[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
                if (knodes[offset[bid]].indices[thid] < knodes_elem) {
                    offset[bid] = knodes[offset[bid]].indices[thid];
                }
            }
        }
        currKnode[bid] = offset[bid];
        #pragma omp task
        kernel_task(bid, maxheight - 1, threadsPerBlock, knodes_elem, currKnode, offset, knodes, keys, records, ans);
    }
}

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task
                kernel_task(bid, maxheight, threadsPerBlock, knodes_elem, currKnode, offset, knodes, keys, records, ans);
            }
        }
    }
}


void kernel_task_j(int bid, int maxheight, int threadsPerBlock, long knodes_elem, long *currKnode, long *offset, long *lastKnode, long *offset_2, knode *knodes, int *start, int *end, int *recstart, int *reclength) {
    if (maxheight == 0) {  // Base case
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
    } else {
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
        kernel_task_j(bid, maxheight - 1, threadsPerBlock, knodes_elem, currKnode, offset, lastKnode, offset_2, knodes, start, end, recstart, reclength);
    }
}

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task
                kernel_task_j(bid, maxheight, threadsPerBlock, knodes_elem, currKnode, offset, lastKnode, offset_2, knodes, start, end, recstart, reclength);
            }
        }
    }
}
