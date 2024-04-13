/*
 * The computation is parallelized using OpenMP's target parallel for directive, distributing the workload across available threads.
 * Memory mapping directives ensure proper data sharing and synchronization between threads.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
    int i;
    #pragma omp target parallel for map(to: N, X[0:N]) map(tofrom: Y[0:N])
    for (i = 0; i < N; ++i) {
        Y[i] += a * X[i];
    }
}
