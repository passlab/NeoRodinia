/*
 * Level 1: Basic Parallel Execution (P1)
 * Description: This is a simple parallelization using parallel for for basic CPU parallel execution.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        Y[i] += X[i];  // Vector addition
}

