/*
 * Level 1: Basic Tasking (P1)
 * In this version, we use basic tasking to split the B+ tree traversal into tasks for each query (bid). Each task processes a query independently, which helps in better workload distribution when there are multiple queries.
 * Description: This version uses basic tasking, where each query (bid) is processed independently as a task. The firstprivate(bid) ensures that each task has its own copy of bid.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task firstprivate(bid)
                {
                    for (int i = 0; i < maxheight; i++) {
                        for (int thid = 0; thid < threadsPerBlock; thid++) {
                            if ((knodes[currKnode[bid]].keys[thid] <= keys[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
                                if (knodes[offset[bid]].indices[thid] < knodes_elem) {
                                    offset[bid] = knodes[offset[bid]].indices[thid];
                                }
                            }
                        }
                        currKnode[bid] = offset[bid];
                    }
                    for (int thid = 0; thid < threadsPerBlock; thid++) {
                        if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
                            ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
                        }
                    }
                }
            }
        }
    }
}

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task firstprivate(bid)
                {
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
        }
    }
}
