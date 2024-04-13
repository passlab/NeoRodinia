/*
 * This kernel utilizes dynamic scheduling with a chunk size of 64 for load balancing across threads.
 * With the `schedule(dynamic, 64)` clause, tasks are dynamically assigned to threads, with each thread processing a chunk of 64 iterations at a time. This dynamic scheduling helps to balance the workload more efficiently, especially when the iteration times vary significantly.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    #pragma omp parallel for shared(N, X, Y, a) private(i) schedule(guided, 64)
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
