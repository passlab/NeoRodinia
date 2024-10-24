/*
 * Level 2: SIMD with Memory Alignment (P2)
 * Description: Adds memory alignment with aligned(Y, X: 32) to ensure that the data is properly aligned for SIMD operations.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel for simd aligned(Y, X: 32)
    for (int i = 0; i < N; ++i) {
        Y[i] += X[i];  // Vector addition
    }
}
