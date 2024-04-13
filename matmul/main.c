#include "matmul.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    
    int default_N = 1024;
    int N = (argc > 1) ? atoi(argv[1]) : default_N;

    REAL *A = malloc(sizeof(REAL)*N*N);
    REAL *B = malloc(sizeof(REAL)*N*N);
    REAL *C_base = malloc(sizeof(REAL)*N*N);
    REAL *C_parallel = malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(A, N);
    init(B, N);
    init(C_base, N);
    memcpy(C_parallel, C_base, N * N * sizeof(REAL));
    
    //serial version for checking the correctness
    REAL temp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C_base[i * N + j] = temp;
        }
    }
    
    double elapsed = read_timer();

    matmul_kernel(N, A, B, C_parallel);
     
    elapsed = (read_timer() - elapsed);
    
    if (need_full_report()) {
        printf("======================================================================================================\n");
        printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\t\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("matmul_kernel:\t\t%4f\t%4f\n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed));
        if (need_verify()) {
            printf("Error (compared to base):\n");
            printf("matmul_kernel:\t\t%g\n", check(C_base, C_parallel, N));
        }
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");
    } else {
         printf("%4f\n",elapsed * 1.0e3);
    }

    free(A);
    free(B);
    free(C_base);
    free(C_parallel);
    return 0;
}
