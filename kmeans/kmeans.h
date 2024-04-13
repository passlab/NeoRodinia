#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include "utils.h"
#include <limits.h>
#include <float.h>
#include <unistd.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define NUM_TEAMS 1024
#define TEAM_SIZE 256

/* Declarations */
int cluster(int, int, float**, int, float, float***);
float **kmeans_clustering(float**, int, int, int, float, int*);
float euclid_dist_2(float*, float*, int);
int find_nearest_point(float*, int, float**, int);
