/*
 * Level 3: Advanced SIMD with SIMD Length Control (P3)
 * Description: Uses simdlen(8) to explicitly control the SIMD vector length for optimized SIMD execution.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        #pragma omp simd simdlen(8) aligned(C, A: 32)
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
