/*
 * Level 2: Optimized Parallel Execution (P2)
 * Description: Adds schedule(dynamic) to optimize the load balancing of the sum operation when the workload might be irregular.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+: result)
    for (int i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}

