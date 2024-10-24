/*
 * Level 1: Basic Tasking (P1)
 * Description: Uses tasking to divide the sum into smaller tasks. Each task computes a partial sum and contributes to the total sum using a critical section.
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
            for (int i = 0; i < N; i += 100) { // Break the array into chunks
                #pragma omp task firstprivate(i) shared(result)
                {
                    REAL partial_sum = 0.0;
                    for (int j = i; j < i + 100 && j < N; j++) {
                        partial_sum += X[j];
                    }
                    #pragma omp critical
                    result += partial_sum;
                }
            }
        }
    }
    return result;
}
