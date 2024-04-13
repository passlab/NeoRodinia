/*
 * Based on P1, P2 emplements the dynamic scheduling strategy employed with the schedule(guided, 64) clause.
 * By using a guided scheduling policy, the workload is dynamically distributed among the threads with a chunk size of 64.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    #pragma omp parallel for shared(N, A, B, C) private(i, j, temp) schedule(guided, 64)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
