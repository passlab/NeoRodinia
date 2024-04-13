/*
 * The `omp target parallel for` pragma is used to distribute the iterations of the inner loop across multiple threads.
 * The `map` clauses specify the data mappings for the variables used in the parallel region.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    
    for (i = 0; i < size; i++) {
    #pragma omp target parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp target parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

