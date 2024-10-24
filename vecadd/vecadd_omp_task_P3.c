/*
 * Level 3: Recursive Tasking (P3)
 * Description: Implements recursive tasking to break the vector addition into smaller chunks and process them recursively.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_task(int start, int end, REAL *Y, REAL *X) {
    if (end - start <= 1000) {  // Base case
        for (int i = start; i < end; i++) {
            Y[i] += X[i];  // Vector addition
        }
    } else {
        int mid = (start + end) / 2;
        #pragma omp task
        vecadd_task(start, mid, Y, X);
        #pragma omp task
        vecadd_task(mid, end, Y, X);
        #pragma omp taskwait
    }
}

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp parallel
    {
        #pragma omp single
        vecadd_task(0, N, Y, X);
    }
}
