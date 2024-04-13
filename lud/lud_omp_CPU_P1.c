/*
 * The `#pragma omp parallel for` directive is used to parallelize both loops.
 * The first inner loop (loop over `j`) updates the elements of the current row (`i`) to form the upper triangular matrix (`U`).
 * The second inner loop (loop over `k`) performs the necessary subtraction to compute the updated values for each element in the current row (`i`).
 * Both inner loops can potentially benefit from parallelization, as the iterations within each loop are independent of each other (data parallelism).
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
