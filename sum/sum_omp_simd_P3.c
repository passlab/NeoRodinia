/*
 * Level 3: Advanced SIMD with SIMD Length Control (P3)
 * Description: Uses simdlen(8) to explicitly control the SIMD vector length for advanced SIMD hardware optimization, ensuring memory alignment for further efficiency.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel for simd simdlen(8) aligned(X: 32) reduction(+: result)
    for (int i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
