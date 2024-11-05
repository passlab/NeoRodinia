/*
 * Level 2: Optimized Parallel Execution with Dynamic Scheduling (P2)
 * This kernel includes a `schedule(dynamic, 64)` clause. This clause specifies that the loop iterations should be dynamically scheduled with a chunk size of 64 iterations.
 * Using dynamic scheduling can improve load balancing, especially if the work per iteration varies significantly.
 * It allows the OpenMP runtime to distribute chunks of iterations dynamically among the available threads, potentially reducing idle time and improving overall parallel efficiency.
 * kernel_k is for option 'k' to find the Kth element in the B+ tree,
 * kernel_j is for option 'j' to find a range of elements in the B+ tree.
 * The user will specify which operation they want to perform and only call one of them.
 */
#include "b+tree.h"
#include <omp.h>

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    int i;
    int thid;
    int bid;

    #pragma omp parallel for private (i, thid) schedule(dynamic, 64)
    // process number of queries
    for (bid = 0; bid < count; bid++) {
        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            // process all leaves at each level
            for (thid = 0; thid < threadsPerBlock; thid++) {
                if ((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid + 1] > start[bid])) {
                    // this conditional statement is inserted to avoid crush due to but in original code
                    // "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
                    // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
                    if (knodes[currKnode[bid]].indices[thid] < knodes_elem) {
                        offset[bid] = knodes[currKnode[bid]].indices[thid];
                    }
                }
                if ((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid + 1] > end[bid])) {
                    // this conditional statement is inserted to avoid crush due to but in original code
                    // "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
                    // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
                    if (knodes[lastKnode[bid]].indices[thid] < knodes_elem) {
                        offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
                    }
                }
            }
            // set for next tree level
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
        }
        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the starting record
            if (knodes[currKnode[bid]].keys[thid] == start[bid]) {
                recstart[bid] = knodes[currKnode[bid]].indices[thid];
            }
        }
        // process leaves
        for (thid = 0; thid < threadsPerBlock; thid++) {
            // Find the index of the ending record
            if (knodes[lastKnode[bid]].keys[thid] == end[bid]) {
                reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid] + 1;
            }
        }
    }
}
