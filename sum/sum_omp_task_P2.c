/*
 * Level 2: Taskloop (P2)
 * Description: Uses taskloop with reduction to parallelize the summation, reducing the need for explicit task creation while balancing the load dynamically.
 *
 */

#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop reduction(+: result)
            for (int i = 0; i < N; ++i) {
                result += X[i];
            }
        }
    }
    return result;
}
