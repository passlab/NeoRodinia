/*
 * Baed on P2, P3 implements num_teams and num_threads clauses.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
    int i;
    #pragma omp target teams distribute parallel for map(to: N, X[0:N]) map(tofrom: Y[0:N]) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
    for (i = 0; i < N; ++i) {
        Y[i] += a * X[i];
    }
}
