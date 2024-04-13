/*
 * Compared to P3, the `num_teams(size/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses in P4 dynamically adjust the number of teams and threads based on the size of the input matrices, achieving one element per thread.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int size = N * N;
    int i, j, k;
    #pragma omp target teams distribute parallel for map(to: N, A[0:size], B[0:size]) map(from: C[0:size]) collapse(2) num_teams(size/TEAM_SIZE) num_threads(TEAM_SIZE)
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
