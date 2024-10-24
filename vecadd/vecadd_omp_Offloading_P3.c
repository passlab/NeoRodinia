/*
 * The `num_teams(N/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses in P3 dynamically adjust the number of teams and threads based on the size of the input matrices, achieving one element per thread.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    int i;
    #pragma omp target teams distribute parallel for map(to: N, X[0:N]) map(tofrom: Y[0:N]) num_teams(N/TEAM_SIZE) num_threads(TEAM_SIZE)
    for (int i = 0; i < N; ++i)
        Y[i] += X[i];  // Vector addition
    }
}
