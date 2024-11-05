/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Introduces a grainsize clause to control task granularity, creating tasks for blocks of rows to optimize workload balance.
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(8)  // Specify grainsize to balance task distribution over blocks of rows
            for (int i = 0; i < N; i++) {
                REAL temp = 0.0;
                for (int j = 0; j < N; j++) {
                    temp += A[i * N + j] * B[j];
                }
                C[i] = temp;
            }
        }
    }
}
