/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Uses recursive tasks for hierarchical traversal through the B+ tree, creating a task for each level in the traversal.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_k_recursive(record *records, knode *knodes, long knodes_elem, int order, long maxheight, long *currKnode, long *offset, int *keys, record *ans, int bid, int level, int threadsPerBlock) {
    if (level >= maxheight) {
        for (int thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
            }
        }
        return;
    }

    for (int thid = 0; thid < threadsPerBlock; thid++) {
        if ((knodes[currKnode[bid]].keys[thid] <= keys[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
            if (knodes[offset[bid]].indices[thid] < knodes_elem) {
                offset[bid] = knodes[offset[bid]].indices[thid];
            }
        }
    }
    currKnode[bid] = offset[bid];

    #pragma omp task
    kernel_k_recursive(records, knodes, knodes_elem, order, maxheight, currKnode, offset, keys, ans, bid, level + 1, threadsPerBlock);
}

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bid = 0; bid < count; bid++) {
                #pragma omp task
                kernel_k_recursive(records, knodes, knodes_elem, order, maxheight, currKnode, offset, keys, ans, bid, 0, threadsPerBlock);
            }
        }
    }
}
