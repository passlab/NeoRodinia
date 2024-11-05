/*
 * Level 1: Basic Taskloop (P1)
 * Description: Uses `taskloop` to create parallel tasks for each query, with each task independently processing a query from root to leaf.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int bid = 0; bid < count; bid++) {
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
