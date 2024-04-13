/*
 * Based on P5, P6 replace critical directive with atomic directive.
 * The `#pragma omp atomic` directive ensures that the update to the `total` sum variable is performed atomically, avoiding race conditions when multiple threads attempt to update the variable simultaneously.
 * The `atomic` directive enhances efficiency by reducing the need for intermediate results calculation. This optimization ensures that the updates to the `total` sum variable are handled atomically, minimizing the overhead associated with coordinating and synchronizing concurrent accesses to the variable.
 *
 */
#include "sum.h"
#include <omp.h>

REAL sum_kernel(int N, REAL X[]) {

    REAL total = 0;

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int chunk_size = N / num_threads;
        int thread_num = omp_get_thread_num();

        int start = thread_num*chunk_size;
        int stop = (thread_num+1)*chunk_size;

        if (thread_num == num_threads-1) {
            int remaining = N % num_threads;
            stop += remaining;
        }

        REAL local_result = 0.0;
        int i;
        for (i = start; i < stop; ++i) {
            local_result +=X[i];
        }

        #pragma omp atomic update
        total += local_result;
    }

    return total;
}
