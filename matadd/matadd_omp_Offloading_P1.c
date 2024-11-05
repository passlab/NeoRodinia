/*
 * Level 1: The computation is parallelized using OpenMP's target parallel for directive.
 * Matrix addition is performed in parallel by distributing the workload across available threads.
 * The map directive ensures proper data transfer between host and device.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    #pragma omp target map(to: A[0:N*N]) map(tofrom: C[0:N*N])
    {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + j];  // Matrix addition
            }
        }
    }
}
