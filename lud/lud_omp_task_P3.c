/*
 * Level 3: Recursive Task Decomposition
 * This version manually creates tasks recursively to divide the computation into smaller subtasks.
 * Implements a recursive task decomposition strategy to divide the first loop into smaller subproblems.
 * Recursive tasks are manually created using #pragma omp task, and synchronization is achieved with #pragma omp taskwait.
 * Efficient for large problem sizes as it allows finer control over task granularity and hierarchical execution.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_recursive(float *a, int size, int i, int start, int end) {
    if (end - start <= 16) { // Base case: process directly for small chunks
        int j, k;
        float sum;
        for (j = start; j < end; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
    } else {
        int mid = (start + end) / 2;

        #pragma omp task shared(a)
        lud_recursive(a, size, i, start, mid);

        #pragma omp task shared(a)
        lud_recursive(a, size, i, mid, end);

        #pragma omp taskwait
    }
}

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;

    for (i = 0; i < size; i++) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                lud_recursive(a, size, i, i, size);

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
