/*
 * Level 2: Dynamic Scheduling
 * Builds on Version 1 but adds dynamic scheduling with a chunk size of 64.
 * The schedule(dynamic, 64) clause divides the loop into chunks of 64 iterations, which are assigned to threads dynamically as they become available.
 * Dynamic scheduling is particularly effective for uneven workloads or when the computational cost of iterations varies, improving thread utilization.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    for (i = 0; i < size; i++) {
        #pragma omp parallel for private(j, k, sum) shared(a) schedule(dynamic, 64)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp parallel for private(j, k, sum) shared(a) schedule(dynamic, 64)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
