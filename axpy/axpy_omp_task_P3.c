/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Implements recursive task creation for handling more complex task dependencies and breaking the task into smaller sub-tasks.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_task(int start, int end, REAL *Y, REAL *X, REAL a) {
    if (end - start < 1024) {  // Base case
        for (int i = start; i < end; ++i)
            Y[i] += a * X[i];
    } else {
        int mid = (start + end) / 2;
        #pragma omp task
        axpy_task(start, mid, Y, X, a);
        #pragma omp task
        axpy_task(mid, end, Y, X, a);
        #pragma omp taskwait
    }
}

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            axpy_task(0, N, Y, X, a);
        }
    }
}
