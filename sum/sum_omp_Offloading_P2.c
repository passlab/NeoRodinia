/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    #pragma omp target teams distribute parallel for map(to: X[0:N]) map(from: result) reduction(+: result)
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
