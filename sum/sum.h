#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define REAL float
#define NUM_TEAMS 1024
#define TEAM_SIZE 256
#define BLOCK_SIZE 64
#define ThreadsPerBlock 256

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N);

double check(REAL *A, REAL B[], int N);

#ifdef __cplusplus
extern "C" {
#endif
extern REAL sum_kernel(int N, REAL X[]);
#ifdef __cplusplus
}
#endif
