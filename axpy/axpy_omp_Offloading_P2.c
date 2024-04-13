/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 *
 */
#include "axpy.h"
#include <omp.h>

/*openmp offloading */
void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
    int i;
    #pragma omp target teams distribute parallel for map(to: N, X[0:N]) map(tofrom: Y[0:N])
    for (i = 0; i < N; ++i) {
        Y[i] += a * X[i];
    }
}
