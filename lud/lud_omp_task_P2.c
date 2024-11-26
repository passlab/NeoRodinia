/*
 * Level 2: Taskloop with Grainsize Control
 * This version adds the grainsize clause to control the size of tasks for better load balancing.
 * Introduces the grainsize clause to control the size of tasks created from the loop iterations.
 * Improves load balancing by ensuring that each task processes a manageable chunk of iterations.
 *  Useful for large matrices, where the overhead of too many small tasks can degrade performance.
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
                #pragma omp taskloop private(j, k, sum) shared(a) grainsize(16)
                for (j = i; j < size; j++) {
                    sum = a[i * size + j];
                    for (k = 0; k < i; k++)
                        sum -= a[i * size + k] * a[k * size + j];
                    a[i * size + j] = sum;
                }

                #pragma omp taskloop private(j, k, sum) shared(a) grainsize(16)
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
