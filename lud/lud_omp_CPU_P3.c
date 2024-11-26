/*
 * Level 3: Dynamic Scheduling with Nested Parallelism
 * Adds nested parallelism for the inner loops, where #pragma omp parallel for is applied to the loops over k.
 * Uses reduction(-:sum) to safely handle partial updates to sum in the inner loop.
 * The inner loop also uses dynamic scheduling with a smaller chunk size (dynamic, 16) to optimize fine-grained parallelism.
 * Nested parallelism can further improve performance on systems with a large number of cores by utilizing threads for both the main decomposition loops and the inner computations.
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
            #pragma omp parallel for reduction(-:sum) schedule(dynamic, 16)
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp parallel for private(j, k, sum) shared(a) schedule(dynamic, 64)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            #pragma omp parallel for reduction(-:sum) schedule(dynamic, 16)
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
