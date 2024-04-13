#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include "utils.h"

#define OPEN

#define NUM_TEAMS 128
#define TEAM_SIZE 128
#define MAXBLOCKSIZE 512
#define BLOCK_SIZE_XY 4

void InitProblemOnce(char *filename);
void InitPerRun(float *m, int Size);

void BackSub(float * finalVec, float *b, float *a, int Size);
void Fan1(float *m, float *a, int Size, int t);
void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void create_matrix(float *m, int size);
double check(float *A, float B[], int N);

#ifdef __cplusplus
extern "C" {
#endif
extern void ForwardSub(int Size, float *a, float *b, float *m);
#ifdef __cplusplus
}
#endif
