/*
 * Level 3: The `num_teams(N/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses dynamically adjust
 * the number of teams and threads based on matrix size, aiming for one element per thread.
 * This version provides more control over the distribution of threads across the GPU.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp target teams distribute parallel for map(to: A[0:N*N]) map(tofrom: C[0:N*N]) num_teams(N/TEAM_SIZE) num_threads(TEAM_SIZE)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
