/*
 * Level 2: Optimized Parallel Execution (P2)
 * Description: Collapses the i and j loops to improve the granularity of parallelism, distributing more work among threads.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j, k;
    REAL temp;
    #pragma omp parallel for collapse(2) private(k, temp)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0;
            for (k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
    }
}

