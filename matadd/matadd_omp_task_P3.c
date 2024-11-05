/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Uses recursive task creation to divide the matrix into row segments, allowing for flexible hierarchical parallelism.
 */
#include "matadd.h"
#include <omp.h>

void matadd_task(int start, int end, int N, REAL *C, REAL *A) {
    if (end - start < 16) {  // Base case: perform addition directly for small row range
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + j];
            }
        }
    } else {
        int mid = (start + end) / 2;
        #pragma omp task
        matadd_task(start, mid, N, C, A);
        #pragma omp task
        matadd_task(mid, end, N, C, A);
        #pragma omp taskwait
    }
}

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            matadd_task(0, N, N, C, A);
        }
    }
}
