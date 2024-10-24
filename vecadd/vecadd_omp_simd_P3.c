/*
 * Level 3: Advanced SIMD with SIMD Length Control (P3)
 * Description: Uses simdlen(8) to explicitly control the SIMD vector length for optimized SIMD execution.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel for simd simdlen(8) aligned(Y, X: 32)
    for (int i = 0; i < N; ++i) {
        Y[i] += X[i];  // Vector addition
    }
}
