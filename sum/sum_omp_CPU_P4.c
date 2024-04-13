/*
 * P4 uses combined directive `parallel for simd reduction(+:result)`
 * This directive combines both parallelization (`parallel for`) and vectorization (`simd`) using OpenMP. It distributes the loop iterations across multiple threads for concurrent execution and utilizes SIMD instructions for vectorized computation.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp parallel for simd reduction(+:result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
