/*
 * Compared to P3, the `num_teams(size/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses in P4 dynamically adjust the number of teams and threads based on the size of the input matrices, achieving one element per thread.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp target teams distribute parallel for map(to: X[0:N]) map(from: result) num_teams(N/TEAM_SIZE) num_threads(TEAM_SIZE) reduction(+: result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
