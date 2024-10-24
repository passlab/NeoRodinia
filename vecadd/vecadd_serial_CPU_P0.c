/*
 * Serial Version
 *
 */
#include "vecadd.h"

void vecadd_kernel(int N, REAL *Y, REAL *X) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += X[i];  // Vector addition
}
