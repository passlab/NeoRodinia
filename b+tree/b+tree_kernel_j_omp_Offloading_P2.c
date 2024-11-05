/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 * kernel_k is for option 'k' to find the Kth element in the B+ tree,
 * kernel_j is for option 'j' to find a range of elements in the B+ tree.
 * The user will specify which operation they want to perform and only call one of them.
 */
#include "b+tree.h"

void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock) {
    int i;
    int thid;
    int bid;
    #pragma omp target teams distribute parallel for private(i, thid) map(to : currKnode [0:count], offset [0:count], start [0:count], end [0:count], lastKnode [0:count], offset_2 [0:count]) map(tofrom : recstart [0:count], reclength [0:count]) map(to : knodes [0:knodes_elem], threadsPerBlock)
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
