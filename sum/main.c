#include "sum.h"
#include "utils.h"

int main(int argc, char *argv[]) {

    int default_N = 10240000;
    int N = (argc > 1) ? atoi(argv[1]) : default_N;

    REAL *X = malloc(sizeof(REAL) * N);

    srand48((1 << 12));
    init(X, N);

    //Serial version used for checking the correctness
    REAL result_serial = 0.0;
    for (int i = 0; i < N; ++i) result_serial += X[i];
   
    double elapsed = read_timer();

    REAL result_parallel = sum_kernel(N, X);
    
    elapsed = (read_timer() - elapsed);
    
    if (need_full_report()) {
        printf("======================================================================================================\n");
        printf("\tSum %d numbers\n", N);
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\t\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("axpy_kernel:\t\t%4f\t%4f \n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed));
        if (need_verify()) {
            printf("Error (compared to base):\n");
            printf("axpy_kernel:\t\t%g\n", result_serial-result_parallel);
        }
    } else {
        printf("%4f\n",elapsed * 1.0e3);
    }

    free(X);

    return 0;
}
