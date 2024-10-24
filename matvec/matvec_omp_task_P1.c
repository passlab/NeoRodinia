/*
 * Level 1: Basic Task Creation (P1)
 * Description: Creates tasks for each row of the result vector C, allowing for dynamic scheduling of matrix-vector operations.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    REAL temp = 0.0;
                    for (int j = 0; j < N; j++) {
                        temp += A[i * N + j] * B[j];
                    }
                    C[i] = temp;
                }
            }
        }
    }
}
