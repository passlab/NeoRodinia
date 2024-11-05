/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses taskloop to parallelize the summation, where each task computes a partial sum. Results are combined in a single-threaded reduction step.
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;

    #pragma omp parallel
    {
        REAL partial_sum = 0.0;
        #pragma omp single
        {
            #pragma omp taskloop reduction(+:partial_sum)
            for (int i = 0; i < N; ++i) {
                partial_sum += X[i];
            }
        }
        #pragma omp atomic
        result += partial_sum;
    }

    return result;
}
