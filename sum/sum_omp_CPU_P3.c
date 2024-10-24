/*
 * Level 3: Nested Parallelism (P3)
 * Description: Uses nested parallelism for more complex scenarios, maintaining a parallel for loop with reduction.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+: result)
        for (int i = 0; i < N; ++i) {
            result += X[i];
        }
    }
    return result;
}
