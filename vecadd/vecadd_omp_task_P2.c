/*
 * Level 2: Taskloop (P2)
 * Description: Uses taskloop to automatically distribute the work among tasks.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < N; ++i) {
                Y[i] += X[i];  // Vector addition
            }
        }
    }
}
