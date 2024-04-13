/*
 * This benchmark implements the LU decomposition of a square matrix `a` of size `size x size` using OpenMP for parallelization.
 * The LU decomposition factors a matrix into the product of a lower triangular matrix (L) and an upper triangular matrix (U).
 * The algorithm iterates over each row of the matrix, updating the elements of the matrix to form the L and U matrices. 
 *
 */
#include "lud.h"
#include "utils.h"

int main(int argc, char *argv[]) {

    int matrix_dim = 32; /* default size */
    int opt, option_index = 0;
    func_ret_t ret;
    const char *input_file = NULL;
    float *m, *mm;

    while ((opt = getopt_long(argc, argv, "::s:i:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                input_file = optarg;
                break;
            case 's':
                matrix_dim = atoi(optarg);
                if (need_full_report()) printf("Generate input matrix internally, size = %d\n", matrix_dim);
                break;
            case '?':
                fprintf(stderr, "invalid option\n");
                break;
            case ':':
                fprintf(stderr, "missing argument\n");
                break;
            default:
                fprintf(stderr, "Usage: %s [-s matrix_size|-i input_file]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if ((optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (input_file) {
        if (need_full_report()) printf("Reading matrix from file %s\n", input_file);
        ret = create_matrix_from_file(&m, input_file, &matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix from file %s\n", input_file);
            exit(EXIT_FAILURE);
        }
    } else if (matrix_dim) {
        if (need_full_report()) printf("Creating matrix internally size=%d\n", matrix_dim);
        ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
            exit(EXIT_FAILURE);
        }
    } else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    if (need_full_report() && need_verify()) {
        printf("Before LUD\n");
        /* print_matrix(m, matrix_dim); */
        matrix_duplicate(m, &mm, matrix_dim);
    }

    double elapsed = read_timer_ms();
    lud_kernel(m, matrix_dim);
    elapsed = (read_timer_ms() - elapsed);

    if (need_full_report()) {
        printf("====================================================================="
               "=================================\n");
        printf("Time consumed(ms): %lf\n", elapsed);
        printf("------------------------------------------------------------------------------------------------------\n");
        if (need_verify()) {
            printf("After LUD\n");
            /* print_matrix(m, matrix_dim); */
            printf(">>>Verify<<<<\n");
            lud_verify(mm, m, matrix_dim);
            free(mm);
        }
        printf("-------------------------------------------------------------------"
               "-----------------------------------\n");
    } else {
        printf("%4f\n", elapsed);
    }

    free(m);

    return EXIT_SUCCESS;
}
