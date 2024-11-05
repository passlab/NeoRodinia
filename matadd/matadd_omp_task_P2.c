/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Adds a grainsize clause to control task granularity, allowing blocks of rows to be processed in each task.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(8)  // Specify grainsize for balanced workload distribution
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += A[i * N + j];
                }
            }
        }
    }
}
