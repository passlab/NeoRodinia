/*
 * Level 1: Basic Parallelization Using #pragma omp parallel for
 * The Fan1 and Fan2 loops are parallelized using the #pragma omp parallel for directive.
 * Work is distributed among threads using the default static scheduling policy.
 * This version allows multiple threads to compute parts of the multiplier matrix (Fan1) and update the LUD matrix (Fan2) simultaneously.
 * Each thread operates independently on its assigned data, with private variables for loop indices (i and j) and shared access to the arrays.
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
void Fan1(float *m, float *a, int Size, int t)
{
    int i;
    #pragma omp parallel for private(i) shared(a,m)
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
    #pragma omp parallel for private(i,j) shared(a,m)
    for (i = 0; i < Size - 1 - t; i++) {
        for (j = 0; j < Size - t; j++)
            a[Size * (i + 1 + t) + (j + t)] -= m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
    }
    #pragma omp parallel for private(i) shared(b,m)
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
