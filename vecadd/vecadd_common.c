#include "vecadd.h"

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N) {
  for (int i = 0; i < N; i++)
    A[i] = (double)drand48();
}

double check(REAL *A, REAL B[], int N) {
  double sum = 0.0;
  for (int i = 0; i < N; i++)
    sum += A[i] - B[i];
  return sum;
}
