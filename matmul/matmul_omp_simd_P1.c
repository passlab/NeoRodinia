/*
 * Level 1: Basic SIMD Vectorization (P1)
 * Description: The innermost loop (for k) is vectorized using the simd directive to enable automatic vectorization for basic SIMD processing.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0;
            #pragma omp parallel for simd private(j, k)
            for (k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
    }
}

