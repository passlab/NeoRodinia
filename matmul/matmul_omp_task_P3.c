/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Implements recursive task creation, splitting the matrix multiplication into smaller subtasks until a base case is reached, suitable for handling large workloads with complex dependencies.
 *
 */
#include "matmul.h"
#include <omp.h>

void matmul_task(int i_start, int i_end, int j_start, int j_end, int N, REAL *A, REAL *B, REAL *C) {
    if ((i_end - i_start) * (j_end - j_start) <= 256) { // Base case
        for (int i = i_start; i < i_end; i++) {
            for (int j = j_start; j < j_end; j++) {
                REAL temp = 0;
                for (int k = 0; k < N; k++) {
                    temp += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = temp;
            }
        }
    } else {
        int i_mid = (i_start + i_end) / 2;
        int j_mid = (j_start + j_end) / 2;
        #pragma omp task
        matmul_task(i_start, i_mid, j_start, j_mid, N, A, B, C);
        #pragma omp task
        matmul_task(i_mid, i_end, j_start, j_mid, N, A, B, C);
        #pragma omp task
        matmul_task(i_start, i_mid, j_mid, j_end, N, A, B, C);
        #pragma omp task
        matmul_task(i_mid, i_end, j_mid, j_end, N, A, B, C);
        #pragma omp taskwait
    }
}

void matmul_kernel(int N, REAL *A, REAL *B, REAL *C) {
    #pragma omp parallel
    {
        #pragma omp single
        matmul_task(0, N, 0, N, N, A, B, C);
    }
}
