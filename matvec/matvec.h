#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL float
#define NUM_TEAMS 1024
#define TEAM_SIZE 256

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N);

double check(REAL *A, REAL B[], int N);
void matvec_serial(int N, REAL *A, REAL *B, REAL *C);

#ifdef __cplusplus
extern "C" {
#endif
extern void matvec_kernel(int N, REAL *A, REAL *B, REAL *C);
#ifdef __cplusplus
}
#endif
