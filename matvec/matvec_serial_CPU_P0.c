/*
 * Serial Version
 *
 */
#include "matvec.h"

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
  int i, j;
  REAL temp;
  for (i = 0; i < N; i++) {
    temp = 0.0;
    for (j = 0; j < N; j++)
      temp += A[i * N + j] * B[j];
    C[i] = temp;
  }
}
