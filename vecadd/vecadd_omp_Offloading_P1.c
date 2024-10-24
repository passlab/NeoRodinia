/*
 * The computation is parallelized using OpenMP's target parallel for directive, distributing the workload across available threads.
 * Memory mapping directives ensure proper data sharing and synchronization between threads.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
#pragma omp target map(to: X[0:N]) map(tofrom: Y[0:N])
    {
        for (int i = 0; i < N; ++i)
            Y[i] += X[i];  // Vector addition
    }
}
