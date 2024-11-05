/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses taskloop to parallelize the outer loop across tasks. Each task computes a single row of the matrix-vector product independently.
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < N; i++) {
                REAL temp = 0.0;
                for (int j = 0; j < N; j++) {
                    temp += A[i * N + j] * B[j];
                }
                C[i] = temp;
            }
        }
    }
}
