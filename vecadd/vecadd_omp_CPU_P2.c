/*
 * Level 2: Optimized Parallel Execution (P2)
 * Description: Adds dynamic scheduling with schedule(dynamic, 128) to improve load balancing for more complex systems where the workload may vary.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < N; ++i)
        Y[i] += X[i];  // Vector addition
}
