/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Introduces a grainsize clause to control task granularity, optimizing workload balance by grouping iterations into blocks.
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
            #pragma omp taskloop grainsize(128) reduction(+:partial_sum)  // Specify grainsize for balanced workload
            for (int i = 0; i < N; ++i) {
                partial_sum += X[i];
            }
        }
        #pragma omp atomic
        result += partial_sum;
    }

    return result;
}
