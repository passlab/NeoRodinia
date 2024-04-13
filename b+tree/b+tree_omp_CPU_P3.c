/*
 * By using the `#pragma omp simd` directive, the inner loops (loop over `thid`) are optimized for SIMD parallelism, which can result in more efficient use of vector processing units in the CPU.
 * This can lead to significant performance improvements, especially for operations on large matrices.
 * kernel_k is for option 'k' to find the Kth element in the B+ tree,
 * kernel_j is for option 'j' to find a range of elements in the B+ tree.
 * The user will specify which operation they want to perform and only call one of them.
 */
#include "b+tree.h"

void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem) {

    int thid;
    int bid;
    int i;
    #pragma omp parallel for private(i, thid) schedule(dynamic, 64)
    // process number of queries
    for (bid = 0; bid < count; bid++) {
        // process levels of the tree
        for (i = 0; i < maxheight; i++) {
            // process all leaves at each level
            #pragma omp simd
            for (thid = 0; thid < threadsPerBlock; thid++) {
                // if value is between the two keys
                if ((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
                    // this conditional statement is inserted to avoid crush due to but in original code
                    // "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
                    // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
                    if (knodes[offset[bid]].indices[thid] < knodes_elem) {
                        offset[bid] = knodes[offset[bid]].indices[thid];
                    }
                }
            }
            // set for next tree level
            currKnode[bid] = offset[bid];
        }
        // At this point, we have a candidate leaf node which may contain
        // the target record.  Check each key to hopefully find the record
        // process all leaves at each level
        for (thid = 0; thid < threadsPerBlock; thid++) {
            if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
                ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
            }
        }
    }
}

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
            #pragma omp simd
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
