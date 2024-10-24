/*
 * This kernel employs the `parallel for` directive to distribute the workload across multiple threads.
 * The variable `i` is declared outside the parallel region but are marked as private to ensure that each thread has its own copy of this variable.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, const REAL *X, REAL a) {
    #pragma omp simd simdlen(4) aligned(X, Y: 64)
    for (int i = 0; i < N; ++i)
        Y[i] += a * X[i];  // AXPY with optimized SIMD
}
