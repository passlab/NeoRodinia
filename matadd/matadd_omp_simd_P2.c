
/*
 * Level 2: SIMD with Memory Alignment (P2)
 * Description: Adds memory alignment with aligned(C, A: 32) to ensure that the data is properly aligned for SIMD operations.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        #pragma omp simd aligned(C, A: 32)
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
