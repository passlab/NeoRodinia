/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 *
 */
#include "vecadd.h"
#include <omp.h>

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; ++i)
        Y[i] += X[i];  // Vector addition
}
