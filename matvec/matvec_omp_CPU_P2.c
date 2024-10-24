/*
 * Level 2: Optimized Parallel Execution (P2)
 * Description: Uses collapse(2) to parallelize both the outer and inner loops, increasing granularity for large matrices.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    #pragma omp parallel for collapse(2) private(temp)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
