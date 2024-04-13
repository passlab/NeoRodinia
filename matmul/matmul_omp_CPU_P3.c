/*
 * Compared to P2, P3 implements the `collapse(2)` clause, combining the nested loops into a single loop, effectively parallelizing both the `i` and `j` iterations. This can improve parallelization efficiency by reducing loop overhead.
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
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}
