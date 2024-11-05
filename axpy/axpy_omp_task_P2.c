/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Introduces a grainsize clause for task granularity control, optimizing for workload balance.
 * Dependencies can be added if there is potential for dependency between tasks.
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(128)  // Specify grainsize for balanced task distribution
            for (int i = 0; i < N; ++i) {
                Y[i] += a * X[i];
            }
        }
    }
}
