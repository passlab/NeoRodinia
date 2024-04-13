/*
 * This kernel utilizes dynamic scheduling with a chunk size of 64 for load balancing across threads.
 * With the `schedule(dynamic, 64)` clause, tasks are dynamically assigned to threads, with each thread processing a chunk of 64 iterations at a time. This dynamic scheduling helps to balance the workload more efficiently, especially when the iteration times vary significantly.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    #pragma omp parallel for shared(N,A,B,C) private(i,j,k,temp) schedule(dynamic, 64)
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
