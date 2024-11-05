/*
 * Level 2: SIMD with Memory Alignment (P2)
 * Description: Adds memory alignment to ensure that data is properly aligned for efficient SIMD execution.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    for (i = 0; i < N; i++) {
        REAL temp = 0.0;
        #pragma omp parallel for simd aligned(A, B, C: 32)
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
