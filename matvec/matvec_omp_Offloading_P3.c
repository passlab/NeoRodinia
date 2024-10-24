/*
 * Baed on P2, P3 implements collapse, num_teams and num_threads clauses.
 * The `collapse(2)` clause combines the nested loops into a single loop, effectively parallelizing both the `i` and `j` iterations. This can improve parallelization efficiency by reducing loop overhead.
 * The `num_teams(size/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses dynamically adjust the number of teams and threads based on the size of the input matrices, achieving one element per thread.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "matvec.h"
#define TILE_SIZE 64

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    int size = N * N;
    #pragma omp target teams distribute parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N]) num_teams(N / TEAM_SIZE) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
