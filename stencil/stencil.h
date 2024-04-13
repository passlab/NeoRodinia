#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define REAL float
#define NUM_TEAMS 1024
#define TEAM_SIZE 256

#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define PROBLEM 1024

void initialize(int width, int height, REAL *u);

#ifdef __cplusplus
extern "C" {
#endif
extern void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
#ifdef __cplusplus
}
#endif
