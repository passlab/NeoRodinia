/*
 * Level 2: Taskloop Parallelization (P2)
 * Description: Uses taskloop to distribute the work across threads with a collapsed loop for better task management and dynamic load balancing.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    REAL temp = 0;
                    for (int k = 0; k < N; k++) {
                        temp += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = temp;
                }
            }
        }
    }
}
