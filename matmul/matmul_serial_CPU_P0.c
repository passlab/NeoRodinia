/*
 * Serial Version
 *
 */
#include "matmul.h"

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0;
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}
