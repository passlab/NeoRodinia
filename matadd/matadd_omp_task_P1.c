/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses taskloop to parallelize the outer loop. Each task processes one row of the matrix independently.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += A[i * N + j];
                }
            }
        }
    }
}
