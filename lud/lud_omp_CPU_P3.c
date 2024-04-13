/*
 * By using the `#pragma omp simd` directive, the inner loops (loop over `k`) are optimized for SIMD parallelism, which can result in more efficient use of vector processing units in the CPU.
 * This can lead to significant performance improvements, especially for operations on large matrices.
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
            #pragma omp simd
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp parallel for private(j, k, sum) shared(a) schedule(dynamic, 64)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            #pragma omp simd
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
