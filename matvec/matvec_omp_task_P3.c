/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Recursively divides the matrix-vector multiplication into smaller tasks, suitable for complex operations where dynamic load balancing is needed.
 *
 */
#include "matvec.h"
#include <omp.h>

void matvec_task(int start, int end, int N, REAL *A, REAL *B, REAL *C) {
    if (end - start <= 256) { // Base case
        for (int i = start; i < end; i++) {
            REAL temp = 0.0;
            for (int j = 0; j < N; j++) {
                temp += A[i * N + j] * B[j];
            }
            C[i] = temp;
        }
    } else {
        int mid = (start + end) / 2;
        #pragma omp task
        matvec_task(start, mid, N, A, B, C);
        #pragma omp task
        matvec_task(mid, end, N, A, B, C);
        #pragma omp taskwait
    }
}

void matvec_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        matvec_task(0, N, N, A, B, C);
    }
}
