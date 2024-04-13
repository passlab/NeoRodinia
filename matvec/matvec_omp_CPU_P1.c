/*
 * Within the parallel region created by the `parallel for` directive, each thread independently iterates over the rows of matrix `A`.
 * The variables `i`, `j`, and `temp` are marked as private to ensure that each thread has its own copy of these variables.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    #pragma omp parallel for shared(N, A, B, C) private(i, j, temp)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
