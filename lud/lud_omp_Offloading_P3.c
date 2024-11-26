/*
 * Level 3: Optimized Teams and Threads
 * Builds on Level 2 by explicitly specifying the number of teams and threads using num_teams and num_threads clauses.
 * Parameters like size/NUM_TEAMS and TEAM_SIZE are introduced to fine-tune the division of work across GPU resources.
 * Optimizes GPU resource utilization by balancing the workload across teams and threads.
 * Suitable for larger datasets or high-performance GPUs with many cores, ensuring that all resources are utilized effectively.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    
    for (i = 0; i < size; i++) {
    #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum) num_teams(size/NUM_TEAMS) num_threads(TEAM_SIZE)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum) num_teams(size/NUM_TEAMS) num_threads(TEAM_SIZE)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

