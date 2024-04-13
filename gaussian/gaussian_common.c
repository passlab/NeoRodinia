#include "utils.h"
#include "gaussian.h"

// create both matrix and right-hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
    int i, j;
    float lamda = -0.01;
    float coe[2 * size - 1];
    float coe_i = 0.0;

    for (i = 0; i < size; i++) {
        coe_i = 10 * exp(lamda * i);
        j = size - 1 + i;
        coe[j] = coe_i;
        j = size - 1 - i;
        coe[j] = coe_i;
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            m[i * size + j] = coe[size - 1 - i + j];
        }
    }
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multiplier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(float *m, int Size) {
    int i;
    for (i = 0; i < Size * Size; i++)
        *(m + i) = 0.0;
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */
void BackSub(float *finalVec, float *b, float *a, int Size) {
    int i, j;
    for (i = 0; i < Size; i++) {
        finalVec[Size - i - 1] = b[Size - i - 1];
        for (j = 0; j < i; j++) {
            finalVec[Size - i - 1] -= *(a + Size * (Size - i - 1) + (Size - j - 1)) * finalVec[Size - j - 1];
        }
        finalVec[Size - i - 1] = finalVec[Size - i - 1] / *(a + Size * (Size - i - 1) + (Size - i - 1));
    }
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size) {
    int i;
    for (i = 0; i < ary_size; i++) {
        printf("%.2f ", ary[i]);
    }
    printf("\n\n");
}

double check(float *A, float B[], int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += A[i] - B[i];
    return sum;
}
