/*
 * Serial Version
 *
 */
#include "sum.h"

REAL sum_kernel(int N, REAL X[]) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i) {
        result += X[i];
    }
    return result;
}
