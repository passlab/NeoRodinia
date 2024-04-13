/*
 * This kernel employs the `parallel for` directive to distribute the workload across multiple threads.
 * The variables `i`, `j`, and `k` are declared outside the parallel region but are marked as private to ensure that each thread has its own copy of these variables.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    #pragma omp parallel for shared(N,A,B,C) private(i,j,k,temp)
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
