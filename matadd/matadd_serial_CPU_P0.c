/*
 * Serial Version
 *
 */
#include "matadd.h"

void matadd_kernel(int N, REAL *C, REAL *A) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            C[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }
}
