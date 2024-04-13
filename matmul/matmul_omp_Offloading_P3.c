/*
 * Baed on P2, P3 implements collapse, num_teams and num_threads clauses.
 * The `collapse(2)` clause combines the nested loops into a single loop, effectively parallelizing both the `i` and `j` iterations. This can improve parallelization efficiency by reducing loop overhead.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "matmul.h"
#include <omp.h>
void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int size = N * N;
    int i, j, k;
    #pragma omp target teams distribute parallel for map(to: N, A[0:size], B[0:size]) map(from: C[0:size]) collapse(2) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
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
