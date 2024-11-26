/*
 * Level 3: SIMD with Reduction
 * Introduces the reduction clause to handle accumulation operations efficiently in SIMD loops.
 * Ensures correctness when performing reductions, such as summations.
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
    // Vectorize and optimize for SIMD with memory alignment
    #pragma omp simd aligned(m, a: 32) simdlen(8)
    for (int i = 0; i < Size - 1 - t; i++) {
        m[Size * (i + t + 1) + t] = a[Size * (i + t + 1) + t] / a[Size * t + t];
    }
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
void Fan2(float *m, float *a, float *b, int Size, int j1, int t) {
    // Vectorize with reduction for row updates
    for (int i = 0; i < Size - 1 - t; i++) {
        #pragma omp simd aligned(m, a: 32) simdlen(8) reduction(+:sum)
        for (int j = 0; j < Size - t; j++) {
            float sum = 0.0f; // Accumulator for reduction
            sum += m[Size * (i + 1 + t) + t] * a[Size * t + (j + t)];
            a[Size * (i + 1 + t) + (j + t)] -= sum;
        }
    }

    // Align and optimize updates to the vector `b`
    #pragma omp simd aligned(m, b: 32) simdlen(8)
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
