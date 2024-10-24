/*
 * Level 2: Optimized Parallel Execution (P2)
 * Description: Adds dynamic scheduling with schedule(dynamic, 128) to balance the load more efficiently.
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
