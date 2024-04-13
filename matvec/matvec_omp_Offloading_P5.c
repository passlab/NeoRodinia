/*
 * Bese on P4, P5 implements the unrolling.
 * The partial clause specifies that the associated loop is first tiled with a tile size of unroll-factor.
 * Then, the generated tile loop is fully unrolled.
 * It is not suppported in clang-14. Use clang-15 for testing.
 */
#include "matvec.h"
#define TILE_SIZE 64

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    int size = N * N;
    #pragma omp target teams distribute parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N]) num_teams(N / TEAM_SIZE) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        #pragma omp unroll partial(4)
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
