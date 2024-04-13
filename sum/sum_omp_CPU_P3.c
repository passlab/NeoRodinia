/*
 * P3 is a vectorized summation routine utilizing OpenMP SIMD directives.
 * `#pragma omp simd reduction(+:result)`: This directive instructs the compiler to vectorize the loop using SIMD instructions, allowing for parallel execution of the loop iterations.
 * The `reduction(+:result)` clause ensures that each thread maintains its private copy of `result` and aggregates them together after the loop execution.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp simd reduction(+:result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
