/*
 * Level 3: Advanced SIMD with Control Over SIMD Length (P3)
 * Description: This kernel uses the simdlen(8) clause to explicitly control the SIMD vector length, allowing the compiler to optimize vectorization based on the target hardware. Additionally, memory alignment is applied to ensure that the data is processed efficiently.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0;
            #pragma omp parallel for simd simdlen(8) private(j, k) aligned(A, B, C: 32)
            for (k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
    }
}
