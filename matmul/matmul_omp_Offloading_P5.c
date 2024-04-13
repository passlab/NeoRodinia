/*
 * This kernel has additional SIMD (Single Instruction, Multiple Data) parallelization.
 * Within the innermost loop, the `parallel for simd` directive parallelizes the computation of each element in the result matrix `C`. SIMD instructions are used to exploit data-level parallelism, enhancing performance by performing multiple operations simultaneously on different data elements.
 * The `reduction(+:temp)` clause ensures proper synchronization and accumulation of the temporary variable `temp`, which holds the partial sum of the matrix multiplication operation for each matrix `C` element.
 * clang-14 and gcc-11 we used for testing don't support simd on GPU so far.
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int size = N * N;
    int i, j, k;
    #pragma omp target teams distribute parallel for map(to: N, A[0:size], B[0:size]) map(from: C[0:size]) collapse(2) num_teams(size/TEAM_SIZE) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0;
            #pragma omp simd reduction(+:temp)
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}
