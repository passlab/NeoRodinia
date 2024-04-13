/*
 * The computation is parallelized using OpenMP's target parallel for directive, distributing the workload across available threads.
 * Memory mapping directives ensure proper data sharing and synchronization between threads.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int size = N * N;
    int i, j, k;
    #pragma omp target parallel for map(to: N, A[0:size], B[0:size]) map(from: C[0:size])
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0;
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}
