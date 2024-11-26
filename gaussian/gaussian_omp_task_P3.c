/*
 * Level 3 (P3): Recursive Task Creation
 * Implements recursive task creation to divide the problem into smaller subtasks.
 * Uses taskgroup to ensure synchronization between nested tasks.
 *
 */
#include "gaussian.h"

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index. Index i gives the range
 ** which starts from 0 to range-1. The real values of
 ** the index should be adjusted and related to the value
 ** of t which is defined in the ForwardSub().
 **-------------------------------------------------------
 */

// Recursive function to handle tasks at each level
void recursive_fan1(float *m, float *a, int Size, int t, int start, int end) {
    if (start >= end) return;

    int mid = (start + end) / 2;

    #pragma omp task
    {
        m[Size * (mid + t + 1) + t] = a[Size * (mid + t + 1) + t] / a[Size * t + t];
    }

    #pragma omp taskgroup
    {
        recursive_fan1(m, a, Size, t, start, mid);
        recursive_fan1(m, a, Size, t, mid + 1, end);
    }
}

void Fan1(float *m, float *a, int Size, int t) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            recursive_fan1(m, a, Size, t, 0, Size - 1 - t);
        }
    }
}
/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
void Fan2(float *m, float *a, float *b, int Size, int j1, int t) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < Size - 1 - t; i++) {
                #pragma omp task
                for (int j = 0; j < Size - t; j++) {
                    a[Size * (i + 1 + t) + (j + t)] -= m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
                }
            }
            #pragma omp taskgroup
            {
                for (int i = 0; i < Size - 1 - t; i++) {
                    b[i + 1 + t] -= m[Size * (i + 1 + t) + t] * b[t];
                }
            }
        }
    }
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(int Size, float *a, float *b, float *m) {
    int t;

    for (t = 0; t < (Size - 1); t++) {
        Fan1(m, a, Size, t);
        Fan2(m, a, b, Size, Size - t, t);
    }
}
