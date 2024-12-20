/*
 * Level 1: Basic Parallel Execution (P1)
 * Description: Basic parallelization using parallel for, where each thread computes one row of the result vector C.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    #pragma omp parallel for private(j, temp)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
