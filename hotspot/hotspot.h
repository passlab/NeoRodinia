#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE
#define STR_SIZE 256
#define NUM_TEAMS 1024
#define TEAM_SIZE 256

/* Maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* Required precision in degrees */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* Capacitance fitting factor */
#define FACTOR_CHIP 0.5
#define OPEN

using namespace std;

typedef float FLOAT;

/* Chip parameters */
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* Ambient temperature, assuming no package at all */
const FLOAT amb_temp = 80.0;

void usage(int argc, char **argv);
void fatal(const char *s);
void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file);
void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file);
extern void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int grid_rows, int grid_cols, int total_iterations);
