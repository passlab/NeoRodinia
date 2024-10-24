/*
 * Level 1: Basic Parallel Execution (P1)
 * Description: Uses parallel for and the reduction clause to parallelize the summation, allowing each thread to calculate a partial sum, which is then reduced into a final result.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel for reduction(+: result)
    for (int i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
