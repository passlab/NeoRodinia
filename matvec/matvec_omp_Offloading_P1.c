/*
 * This kernel offloads the computation to the GPU device for parallel execution.
 * Within the `target parallel for` directive, the workload is distributed across the available threads on the GPU.
 * The `map` clause is used to explicitly specify the data mapping between the host and the device.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    int size = N * N;
    #pragma omp target parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N])
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
