/*
 * Level 1: Basic Taskloop Parallelism
 * This version uses the #pragma omp taskloop directive to divide the loops into tasks automatically.
 * Uses #pragma omp taskloop for both loops, splitting iterations into tasks automatically.
 * Tasks are created dynamically, and each thread in the parallel region picks up available tasks.
 * A simple way to introduce task-based parallelism without manual task management.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;

    for (i = 0; i < size; i++) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp taskloop private(j, k, sum) shared(a)
                for (j = i; j < size; j++) {
                    sum = a[i * size + j];
                    for (k = 0; k < i; k++)
                        sum -= a[i * size + k] * a[k * size + j];
                    a[i * size + j] = sum;
                }

                #pragma omp taskloop private(j, k, sum) shared(a)
                for (j = i + 1; j < size; j++) {
                    sum = a[j * size + i];
                    for (k = 0; k < i; k++)
                        sum -= a[j * size + k] * a[k * size + i];
                    a[j * size + i] = sum / a[i * size + i];
                }
            }
        }
    }
}

