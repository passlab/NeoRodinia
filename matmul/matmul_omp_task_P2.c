/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Introduces a grainsize clause for the taskloop, allowing control over task granularity by grouping rows into blocks to improve workload balance.
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(4)  // Specify grainsize for block of rows to balance task distribution
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    REAL temp = 0.0;
                    for (int k = 0; k < N; k++) {
                        temp += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = temp;
                }
            }
        }
    }
}
