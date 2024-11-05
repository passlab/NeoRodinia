/*
 * Level 1: Basic SIMD Vectorization (P1)
 * Description: Uses SIMD vectorization in the inner loop to parallelize matrix addition at the instruction level.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
