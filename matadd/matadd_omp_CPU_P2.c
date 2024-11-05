/*
 * Level 2: Chunked Parallel For with Dynamic Scheduling (P2)
 * Description: Adds dynamic scheduling with a specified chunk size to balance workload. Useful when row computation may vary in cost.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel for schedule(dynamic, 8)  // Dynamically assign chunks of rows to threads
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];
        }
    }
}
