/*
 * Level 1: Basic Parallelization
 * Introduces basic parallelization using #pragma omp parallel for.
 * Each iteration of the outer loop (LU decomposition phase) is split into two parallel loops: one for updating the upper triangular matrix and the other for updating the lower triangular matrix.
 * The private and shared clauses manage thread safety.
 * The workload distribution is static, leaving the loop partitioning to the compiler's default behavior.
 *
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    for (i = 0; i < size; i++) {
        #pragma omp parallel for private(j, k, sum) shared(a)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp parallel for private(j, k, sum) shared(a)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}
