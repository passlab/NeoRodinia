/*
 * Level 3: Nested Parallelism (P3)
 * Description: Uses nested parallelism for more complex parallel execution strategies.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i)
            Y[i] += X[i];  // Vector addition
    }
}
