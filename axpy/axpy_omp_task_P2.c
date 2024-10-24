/*
 * Level 2: Advanced Taskloop with Dependencies (P2)
 * Description: Adds depend clauses to manage task dependencies and improve scheduling in more complex cases.
 *
 */
#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop depend(inout: Y[0:N])
            for (int i = 0; i < N; ++i)
                Y[i] += a * X[i];
        }
    }
}
