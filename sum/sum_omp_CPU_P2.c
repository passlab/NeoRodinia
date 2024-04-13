/*
 * P2 introduces guided scheduling.
 * The code achieves parallel execution with dynamic adjustment of the chunk size based on runtime behavior.
 * This can lead to better load balancing and improved performance compared to static scheduling, especially for cases where workload distribution across iterations varies significantly.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp parallel for reduction(+:result) schedule(guided, 64)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
