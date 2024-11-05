/*
 * Level 1: Simple Parallelization with Parallel For (P1)
 * Description: Uses a parallel for loop to distribute rows across threads. Each thread computes the addition for a subset of rows.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];
        }
    }
}
