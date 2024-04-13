/*
 * Based on P2, P3 implements the `simd` directive is indicate that the loop can be vectorized.
 * The `reduction(+ : temp)` clause ensures proper synchronization and accumulation of the temporary variable `temp`, which holds the partial sum of the dot product.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    #pragma omp parallel for shared(N, A, B, C) private(i, j, temp) schedule(guided, 64)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        #pragma omp simd reduction(+ : temp)
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
