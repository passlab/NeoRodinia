/*
 * This kernel includes a `schedule(dynamic, 64)` clause. This clause specifies that the loop iterations should be dynamically scheduled with a chunk size of 64 iterations.
 * Using dynamic scheduling can improve load balancing, especially if the work per iteration varies significantly.
 * It allows the OpenMP runtime to distribute chunks of iterations dynamically among the available threads, potentially reducing idle time and improving overall parallel efficiency.
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
