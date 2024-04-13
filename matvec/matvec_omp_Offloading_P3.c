/*
 * Baed on P2, P3 implements collapse, num_teams and num_threads clauses.
 * The `collapse(2)` clause combines the nested loops into a single loop, effectively parallelizing both the `i` and `j` iterations. This can improve parallelization efficiency by reducing loop overhead.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    int size = N * N;
    #pragma omp target teams distribute parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N]) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
