#include "axpy.h"
#include "utils.h"

int main(int argc, char *argv[]) {

  int default_N = 1024000;

  int N = (argc > 1) ? atoi(argv[1]) : default_N;

  REAL a = 123.456;
  REAL *X = malloc(sizeof(REAL) * N);
  REAL *Y_base = malloc(sizeof(REAL) * N);
  REAL *Y_parallel = malloc(sizeof(REAL) * N);

  srand48((1 << 12));
  init(X, N);
  init(Y_base, N);
  memcpy(Y_parallel, Y_base, N * sizeof(REAL));

  // Serial version used for checking the correctness
  for (int i = 0; i < N; ++i)
    Y_base[i] += a * X[i];

  double elapsed = read_timer();

  axpy_kernel(N, Y_parallel, X, a);

  elapsed = (read_timer() - elapsed);
  if (need_full_report()) {
    printf("==================================================================="
           "===================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N = %d\n", N);
    printf("-------------------------------------------------------------------"
           "-----------------------------------\n");
    printf("-------------------------------------------------------------------"
           "-----------------------------------\n");

    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("axpy_kernel:\t\t%4f\t%4f\n", elapsed * 1.0e3,
           (2.0 * N) / (1.0e6 * elapsed));
    if (need_verify()) {
      printf("Error (compared to base):\n");
      printf("axpy_kernel:\t\t%g\n", check(Y_base, Y_parallel, N));
    }
    printf("-------------------------------------------------------------------"
           "-----------------------------------\n");
  } else {
    printf("%4f\n", elapsed * 1.0e3);
  }

  free(Y_base);
  free(Y_parallel);
  free(X);

  return 0;
}
