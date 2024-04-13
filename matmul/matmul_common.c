#include "matmul.h"

void init(REAL *A, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double check(REAL *A, REAL B[], int N) {
    int i,j;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum += A[i*N+j] - A[i*N+j];
        }
    }
    return sum;
}
