#include "matvec.h"

void init(REAL *A, int N) {
  for (int i = 0; i < N; i++)
    A[i] = (REAL)drand48();
}

double check(REAL *A, REAL B[], int N) {
  double sum = 0.0;
  for (int i = 0; i < N; i++)
    sum += A[i] - B[i];
  return sum;
}

void matvec_serial(int N, REAL *A, REAL *B, REAL *C) {
  int i, j;
  REAL temp;
  for (i = 0; i < N; i++) {
    temp = 0.0;
    for (j = 0; j < N; j++)
      temp += A[i * N + j] * B[j];
    C[i] = temp;
  }
}
