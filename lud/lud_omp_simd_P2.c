/*
 * Level 2: SIMD with Aligned Data
 * This version ensures that data accesses are aligned, enabling more efficient SIMD operations.
 * Introduces the aligned clause in the #pragma omp simd directive to specify that the data is aligned in memory.
 * Aligned data allows the SIMD instructions to operate more efficiently by reducing memory overhead.
 * Improves performance for larger datasets where memory alignment is critical for vectorized execution.
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
            #pragma omp simd aligned(a : 64)
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }

        #pragma omp parallel for shared(a) private(j, k, sum)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            #pragma omp simd aligned(a : 64)
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
