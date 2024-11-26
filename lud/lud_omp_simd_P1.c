/*
 * Level 1: Basic SIMD Implementation
 * This version introduces basic SIMD parallelism using the #pragma omp simd directive to vectorize the inner loops.
 * Adds #pragma omp simd for the innermost loops to enable basic vectorization.
 * Exploits SIMD hardware capabilities to execute multiple iterations of the inner loop in parallel.
 * Focuses on simplicity and quick adoption of SIMD without additional optimizations.
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;

    for (i = 0; i < size; i++) {
        #pragma omp parallel for shared(a) private(j, k, sum)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            #pragma omp simd
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }

        #pragma omp parallel for shared(a) private(j, k, sum)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            #pragma omp simd
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
