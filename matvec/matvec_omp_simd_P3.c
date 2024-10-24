/*
 * Level 3: Advanced SIMD with Control Over SIMD Length (P3)
 * Description: Uses simdlen(8) to explicitly control the SIMD vector length, providing advanced optimization for hardware with specific vector length requirements.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j;
    #pragma omp parallel for simd simdlen(8) private(j) aligned(A, B, C: 32)
    for (i = 0; i < N; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i * N + j] * B[j];
        }
        C[i] = temp;
    }
}
