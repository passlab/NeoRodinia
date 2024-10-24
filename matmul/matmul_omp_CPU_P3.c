/*
 * In P3, the `simd` directive is used within the innermost loop to indicate that the loop can be vectorized using SIMD (Single Instruction, Multiple Data) instructions for further optimization.
 * The `reduction(+:temp)` clause ensures proper synchronization and accumulation of the temporary variable `temp`, which holds the partial sum of the matrix multiplication operation for each element of matrix `C`.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    #pragma omp parallel for shared(N,A,B,C) private(i,j,k,temp) schedule(dynamic, 64) collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0;
            #pragma omp simd reduction(+:temp)
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}

