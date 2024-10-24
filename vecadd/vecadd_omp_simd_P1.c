/*
 * Level 1: Basic SIMD Vectorization (P1)
 * Description: Uses SIMD vectorization to parallelize the vector addition at the instruction level.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i) {
        Y[i] += X[i];  // Vector addition
    }
}
