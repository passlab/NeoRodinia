/*
 * Level 1: Basic Parallel Execution (P1)
 * Description: This is a simple parallelization using parallel for for basic CPU parallel execution.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
