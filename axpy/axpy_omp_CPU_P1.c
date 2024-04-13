/*
 * This kernel employs the `parallel for` directive to distribute the workload across multiple threads.
 * The variable `i` is declared outside the parallel region but are marked as private to ensure that each thread has its own copy of this variable.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    #pragma omp parallel for shared(N, X, Y, a) private(i)
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
