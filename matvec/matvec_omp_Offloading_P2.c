/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    REAL temp;
    int size = N * N;
    #pragma omp target teams distribute parallel for map(to : N, A[0 : size], B[0 : N]) map(from : C[0 : N])
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}
