/*
 * Level 3: Recursive Tasking (P3)
 * Description: Implements recursive tasking, splitting the summation into smaller tasks until a base case is reached, then combining the results from each task.
 *
 */

#include "sum.h"
#include <omp.h>

void sum_task(int start, int end, REAL X[], REAL* result) {
    if (end - start <= 100) {  // Base case
        for (int i = start; i < end; i++) {
            *result += X[i];
        }
    } else {
        REAL left_result = 0.0, right_result = 0.0;
        int mid = (start + end) / 2;
        #pragma omp task shared(left_result)
        sum_task(start, mid, X, &left_result);
        #pragma omp task shared(right_result)
        sum_task(mid, end, X, &right_result);
        #pragma omp taskwait
        *result = left_result + right_result;
    }
}

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        sum_task(0, N, X, &result);
    }
    return result;
}
