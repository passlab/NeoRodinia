/*
 * `#pragma omp parallel for reduction(+:result)`: This directive parallelizes the loop using OpenMP, distributing the loop iterations across multiple threads for concurrent execution.
 * The `reduction(+:result)` clause ensures that each thread maintains its private copy of `result` and aggregates them together after the loop execution.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
