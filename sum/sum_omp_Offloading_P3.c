/*
 * Baed on P2, P3 implements num_teams and num_threads clauses.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp target teams distribute parallel for map(to: X[0:N]) map(from: result) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE) reduction(+: result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
