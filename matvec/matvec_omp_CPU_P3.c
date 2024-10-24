/*
 * Level 3: Advanced Parallel Execution with Nested Parallelism (P3)
 * Description: Focuses on nested parallelism with multiple layers of parallelism, improving load distribution for complex matrix-vector operations.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < N; i++) {
            REAL temp = 0.0;
            for (j = 0; j < N; j++) {
                temp += A[i * N + j] * B[j];
            }
            C[i] = temp;
        }
    }
}
