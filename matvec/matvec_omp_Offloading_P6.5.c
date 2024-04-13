/*
 * P6.5 implements the mannualy tiling.
 *
 */
#include "matvec.h"
#define TILE_SIZE 32

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j, k;
    int size = N * N;
    #pragma omp target teams distribute parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N]) num_teams(N / TEAM_SIZE) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        REAL temp = 0.0;
        for (k = 0; k < N; k += TILE_SIZE) {
            REAL temp_block[TILE_SIZE];
            for (j = 0; j < TILE_SIZE; j++) {
                temp_block[j] = A[i * N + k + j] * B[k + j];
            }
            for (j = 0; j < TILE_SIZE; j++) {
                temp += temp_block[j];
            }
        }
        C[i] = temp;
    }
}
