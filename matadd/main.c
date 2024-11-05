#include "matadd.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    int default_N = 1024;  // Default matrix dimension, assuming N x N

    int N = (argc > 1) ? atoi(argv[1]) : default_N;

    REAL *A = malloc(sizeof(REAL) * N * N);
    REAL *B = malloc(sizeof(REAL) * N * N);
    REAL *C_base = malloc(sizeof(REAL) * N * N);
    REAL *C_parallel = malloc(sizeof(REAL) * N * N);

    srand48((1 << 12));
    init(A, N * N);
    init(B, N * N);
    memcpy(C_base, B, N * N * sizeof(REAL));
    memcpy(C_parallel, B, N * N * sizeof(REAL));

    // Serial version used for checking correctness
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C_base[i * N + j] += A[i * N + j];  // Matrix addition
        }
    }

    double elapsed = read_timer();

    matadd_kernel(N, C_parallel, A);  // Updated kernel for matrix addition

    elapsed = (read_timer() - elapsed);
    if (need_full_report()) {
        printf("==================================================================="
               "===================================\n");
        printf("\tMatrix Addition: C[N][N] = C[N][N] + A[N][N], N = %d\n", N);
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");

        printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
        printf("matrix_add_kernel:\t\t%4f\t%4f\n", elapsed * 1.0e3,
               (2.0 * N * N) / (1.0e6 * elapsed));  // Adjusted MFLOPS formula
        if (need_verify()) {
            printf("Error (compared to base):\n");
            printf("matrix_add_kernel:\t\t%g\n", check(C_base, C_parallel, N * N));
        }
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");
    } else {
        printf("%4f\n", elapsed * 1.0e3);
    }

    free(C_base);
    free(C_parallel);
    free(A);

    return 0;
}
