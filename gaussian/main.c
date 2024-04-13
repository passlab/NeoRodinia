#include "utils.h"
#include "gaussian.h"

int main(int argc, char *argv[]) {
    
    int i,j;
    char flag;
    
    int Size;
    float *a, *b, *finalVec;
    float *m;
    
    FILE *fp;
    float *verify;

    //unsigned int totalKernelTime = 0;
    
    if (argc < 2) {
        printf("Usage: gaussian -f filename / -s size [-q]\n\n");
        printf("-q (quiet) suppresses printing the matrix and result values.\n");
        printf("-f (filename) path of input file\n");
        printf("-s (size) size of matrix. Create matrix and rhs in this program \n");
        printf("The first line of the file contains the dimension of the matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf("The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. (optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6    -0.5    0.7    0.3\n");
        printf("-0.3    -0.9    0.3    0.7\n");
        printf("-0.4    -0.5    -0.3    -0.8\n");
        printf("0.0    -0.1    0.2    0.9\n");
        printf("\n");
        printf("-0.85    -0.68    0.24    -0.53\n");
        printf("\n");
        printf("0.7    0.0    -0.4    -0.5\n");
        exit(0);
    }
    for(i=1;i<argc;i++) {
        if (argv[i][0]=='-') {
            flag = argv[i][1];
            switch (flag) {
                case 's': // platform
                    i++;
                    Size = atoi(argv[i]);
                    if (need_full_report()) printf("Create matrix internally in parse, size = %d \n", Size);
                    
                    a = (float *) malloc(Size * Size * sizeof(float));
                    create_matrix(a, Size);
                    b = (float *) malloc(Size * sizeof(float));
                    for (j =0; j< Size; j++)
                        b[j]=1.0;
                    m = (float *) malloc(Size * Size * sizeof(float));
                    break;
                case 'f':
                    i++;
                    if (need_full_report()) printf("Read file from %s \n", argv[i]);
                    fp = fopen(argv[i], "r");
                    fscanf(fp, "%d", &Size);
                    a = (float *) malloc(Size * Size * sizeof(float));
                    b = (float *) malloc(Size * sizeof(float));
                    m = (float *) malloc(Size * Size * sizeof(float));
                    for (i=0; i<Size; i++) {
                        for (j=0; j<Size; j++) {
                            fscanf(fp, "%f", &a[Size * i + j]);
                        }
                    }
                    for (i=0; i<Size; i++) {
                        fscanf(fp, "%f",  &b[i]);
                    }
                    verify = (float *) malloc(Size * sizeof(float));
                    for (i=0; i<Size; i++) {
                        fscanf(fp, "%f",  &verify[i]);
                    }
                    break;
                case 'q':
                    break;
            }
        }
    }
    InitPerRun(m,Size);
    
    double elapsed = read_timer();
    ForwardSub(Size, a, b, m);
    elapsed = (read_timer() - elapsed);
    
    if (need_full_report()) {
        if(Size <= 6){
            printf("Matrix m is: \n");
            for (i = 0; i < Size; i++) {
                for (j = 0; j < Size; j++) {
                    printf("%8.2f ", m[Size * i + j]);
                }
                printf("\n");
            }
            printf("Matrix a is: \n");
            for (i = 0; i < Size; i++) {
                for (j = 0; j < Size; j++) {
                    printf("%8.2f ", a[Size * i + j]);
                }
                printf("\n");
            }
            printf("Array b is: \n");
            PrintAry(b, Size);
        } else {
            printf("The matrix is too large, only the first 6 lines and columns are printed. \n");
            printf("Matrix m is: \n");
            for (i = 0; i < 6; i++) {
                for (j = 0; j < 6; j++) {
                    printf("%8.2f ", m[Size * i + j]);
                }
                printf("\n");
            }
            printf("Matrix a is: \n");
            for (i = 0; i < 6; i++) {
                for (j = 0; j < 6; j++) {
                    printf("%8.2f ", a[Size * i + j]);
                }
                printf("\n");
            }
            printf("Array b is: \n");
            PrintAry(b, 6);
            
        }
    }
    finalVec = (float *) malloc(Size * sizeof(float));
    BackSub(finalVec, b, a, Size);
    if (need_full_report()) {
        printf("The final solution is: \n");
        PrintAry(finalVec, Size);
        printf("Time for kernels:\t%f sec\n", elapsed * 1.0e3);
        if (need_verify()) {
            if (flag == 'f') {
                printf("Error (compared to base):%g\n", check(verify, finalVec, Size));
            } else {
                printf("Randomly generated test, please check it manually.\n");
            }
        }
    } else {
        printf("%4f\n", elapsed * 1.0e3);
    }
    free(m);
    free(a);
    free(b);
}
