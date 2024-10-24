/*
 * Level 1: Basic SIMD Vectorization (P1)
 * Description: Uses SIMD vectorization along with parallel for and reduction to vectorize the summation operation.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel for simd reduction(+: result)
    for (int i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
