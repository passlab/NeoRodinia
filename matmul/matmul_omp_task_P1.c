/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses task for each row of matrix A. Each task handles the computation for one row of the result matrix.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task firstprivate(i)
                {
                    for (int j = 0; j < N; j++) {
                        REAL temp = 0;
                        for (int k = 0; k < N; k++) {
                            temp += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = temp;
                    }
                }
            }
        }
    }
}
