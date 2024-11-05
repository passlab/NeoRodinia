/*
 * Level 2: SIMD with Memory Alignment (P2)
 * Description: Memory alignment is added using the aligned(A, B, C: 32) clause to ensure that the data is aligned in memory for more efficient vectorized operations. This is important when working with SIMD hardware.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0;
            #pragma omp parallel for simd private(j, k) aligned(A, B, C: 32)
            for (k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
    }
}
