/*
 * This kernel is a parallelized summation routine utilizing OpenMP directives for offloading computations to a target device, likely a GPU.
 * `#pragma omp target parallel for` distributes the loop iterations across multiple threads on the target device for concurrent execution.
 
 * `reduction(+: result)` specifies that a reduction operation should be performed on the `result` variable across all threads.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp target parallel for map(to: X[0:N]) map(from: result) reduction(+: result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
