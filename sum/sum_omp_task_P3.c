/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Implements recursive task creation for summing the array in segments, creating a binary reduction tree for efficient summation.
 */
#include "sum.h"
#include <omp.h>

REAL sum_recursive(int start, int end, REAL X[]) {
    if (end - start < 1024) {  // Base case: compute sum directly for small range
        REAL partial_sum = 0.0;
        for (int i = start; i < end; ++i) {
            partial_sum += X[i];
        }
        return partial_sum;
    } else {
        int mid = (start + end) / 2;
        REAL left_sum, right_sum;
        
        #pragma omp task shared(left_sum)
        left_sum = sum_recursive(start, mid, X);
        
        #pragma omp task shared(right_sum)
        right_sum = sum_recursive(mid, end, X);
        
        #pragma omp taskwait
        return left_sum + right_sum;
    }
}

REAL sum_kernel(int N, REAL X[]) {
    REAL result = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = sum_recursive(0, N, X);
        }
    }

    return result;
}
