/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses taskloop to parallelize the outer loop across tasks. Each task computes a row of the result matrix independently.
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    REAL temp = 0.0;
                    for (int k = 0; k < N; k++) {
                        temp += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = temp;
                }
            }
        }
    }
}
