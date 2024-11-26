/*
 * Level 2 (P2): Grain Size for Fine-Grained Task Control
 * Introduces the grainsize clause in taskloop to explicitly control the size of tasks.
 * This enables better load balancing, especially for uneven workloads.
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
void Fan1(float *m, float *a, int Size, int t) {
    // Use taskloop with grainsize for finer control
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(16)
            for (int i = 0; i < Size - 1 - t; i++) {
                m[Size * (i + t + 1) + t] = a[Size * (i + t + 1) + t] / a[Size * t + t];
            }
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
            #pragma omp taskloop grainsize(16)
            for (int i = 0; i < Size - 1 - t; i++) {
                for (int j = 0; j < Size - t; j++) {
                    a[Size * (i + 1 + t) + (j + t)] -= m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
                }
            }
            #pragma omp taskloop grainsize(16)
            for (int i = 0; i < Size - 1 - t; i++) {
                b[i + 1 + t] -= m[Size * (i + 1 + t) + t] * b[t];
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
