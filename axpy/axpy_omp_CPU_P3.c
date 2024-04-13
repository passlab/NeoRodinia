/*
 * Compared to P2, in P3, the `simd` directive is used to indicate that the loop can be vectorized using SIMD (Single Instruction, Multiple Data) instructions for further optimization.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    #pragma omp parallel for simd shared(N, X, Y, a) private(i)
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
