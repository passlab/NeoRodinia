/*
 * Level 2: SIMD with Memory Alignment (P2)
 * Description: Adds memory alignment with aligned(X: 32) to ensure the array is properly aligned for efficient SIMD operations.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel for simd aligned(X: 32) reduction(+: result)
    for (int i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
