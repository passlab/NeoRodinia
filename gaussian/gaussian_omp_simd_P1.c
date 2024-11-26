/*
 * Level 1: Basic SIMD Vectorization
 * The innermost loops in Fan1 and Fan2 are vectorized using #pragma omp simd.
 * This applies SIMD instructions to multiple data points, enabling data-level parallelism.
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
    // Vectorize the loop to compute the multiplier for each row
    #pragma omp simd
    for (int i = 0; i < Size - 1 - t; i++) {
        m[Size * (i + t + 1) + t] = a[Size * (i + t + 1) + t] / a[Size * t + t];
    }
}
/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
void Fan2(float *m, float *a, float *b, int Size, int j1, int t) {
    // Parallelize the row-wise updates
    for (int i = 0; i < Size - 1 - t; i++) {
        // Vectorize the column-wise computation
        #pragma omp simd
        for (int j = 0; j < Size - t; j++) {
            a[Size * (i + 1 + t) + (j + t)] -= m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
        }
    }

    // Update the right-hand side vector
    #pragma omp simd
    for (int i = 0; i < Size - 1 - t; i++) {
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
