/*
 * Baed on P2, P3 implements num_teams and num_threads clauses.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    
    for (i = 0; i < size; i++) {
    #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

