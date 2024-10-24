/*
 * Level 1: Basic Tasking (P1)
 * Description: Uses tasking to divide the vector addition into chunks. Each task processes a part of the vector independently.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i += 1000) {
                #pragma omp task firstprivate(i)
                {
                    for (int j = i; j < i + 1000 && j < N; j++) {
                        Y[j] += X[j];  // Vector addition
                    }
                }
            }
        }
    }
}
