#include "matvec.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    
    int default_N = 10240;
    int N = (argc > 1) ? atoi(argv[1]) : default_N;

    REAL *A = malloc(sizeof(REAL) * N * N);
    REAL *B = malloc(sizeof(REAL) * N);
    REAL *C_base = malloc(sizeof(REAL) * N);
    REAL *C_parallel = malloc(sizeof(REAL) * N);
    
    srand48((1 << 12));
    init(A, N * N);
    init(B, N);
    init(C_base, N);
    memcpy(C_parallel, C_base, N * sizeof(REAL));
    
    matvec_serial(N, A, B, C_base);

    double elapsed = read_timer();

    matvec_kernel(N, A, B, C_parallel);
         
    elapsed = (read_timer() - elapsed);
    
    if (need_full_report()) {
        printf("====================================================================="
        "=================================\n");
        printf("\tMatrix Vector Multiplication: A[N][N] * B[N] = C[N], N=%d\n", N);
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\t\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("matvec_kernel:\t\t%4f\t%4f\n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed));
        if (need_verify()) {
            printf("Error (compared to base):\n");
            printf("matmul_kernel:\t\t%g\n", check(C_base, C_parallel, N));
        }
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");
    } else {
            printf("%4f\n",elapsed * 1.0e3);
    }
    return 0;
}


