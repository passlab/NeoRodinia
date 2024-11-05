/*
 * Level 1: Basic Task Creation (P1)
 * Description: Uses taskloop to parallelize tasks, distributing iterations across tasks.
 * Suitable for independent task execution with minimal dependency overhead.
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop
            for (int i = 0; i < N; ++i) {
                Y[i] += a * X[i];
            }
        }
    }
}
