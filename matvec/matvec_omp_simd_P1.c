/*
 * Level 1: Basic SIMD Vectorization (P1)
 * Description: Basic SIMD vectorization of the inner loop to enable automatic vectorization for the matrix-vector multiplication.
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    #pragma omp simd
    for (i = 0; i < N; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
