/*
 * Level 3: Dynamic Scheduling with Collapsed Loops
 * This version builds on the dynamic scheduling approach and introduces the collapse clause to parallelize nested loops in the Fan2 function. It aims to maximize thread utilization and further enhance load balancing by flattening the loop hierarchy.
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
void Fan1(float *m, float *a, int Size, int t)
{
    int i;
    #pragma omp parallel for private(i) shared(a,m) schedule(dynamic, 64)
    for (i = 0; i < Size - 1 - t; i++)
        m[Size * (i + t + 1) + t] = a[Size * (i + t + 1) + t] / a[Size * t + t];
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
void Fan2(float *m, float *a, float *b, int Size, int j1, int t)
{
    int i, j;
    #pragma omp parallel for private(i,j) shared(a,m) schedule(dynamic, 64) collapse(2)
    for (i = 0; i < Size - 1 - t; i++) {
        for (j = 0; j < Size - t; j++)
            a[Size * (i + 1 + t) + (j + t)] -= m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
    }
    #pragma omp parallel for private(i) shared(b,m) schedule(dynamic, 64)
    for (i = 0; i < Size - 1 - t; i++) {
        b[i + 1 + t] -= m[Size * (i + 1 + t) + t] * b[t];
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
