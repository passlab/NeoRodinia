/*
 * Level 3: SIMD with simdlen Optimization
 * This version uses the simdlen clause to specify the preferred SIMD width, optimizing for the target architecture.
 * Uses the simdlen clause to specify the preferred SIMD vector length (simdlen(8)), which corresponds to 8 elements per vector register.
 * Combines simdlen with aligned for optimal memory and register usage.
 * Targets hardware-specific optimizations, leveraging architecture-specific capabilities to maximize SIMD efficiency.
 *
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
            #pragma omp simd simdlen(8) aligned(a : 64)
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }

        #pragma omp parallel for shared(a) private(j, k, sum)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            #pragma omp simd simdlen(8) aligned(a : 64)
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
