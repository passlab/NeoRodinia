/*
 * Level 2: Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes rows among teams of threads, and each team further distributes columns among threads.
 * This approach can provide better load balancing and resource utilization.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
