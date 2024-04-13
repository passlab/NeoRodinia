/*
 * P5 is a parallelized summation routine utilizing OpenMP directives. It divides the summation task among multiple threads, each calculating a portion of the total sum.
 *  Inside the parallel region, each thread calculates a portion of the total sum. The array `X` is divided into chunks, and each thread is responsible for summing a portion of the array.
 * The number of threads (`num_threads`) and the chunk size (`chunk_size`) are calculated using OpenMP functions `omp_get_num_threads()` and division of `N`.
 * Each thread determines its start and stop indices (`start` and `stop`) based on its thread number and chunk size.
 * If it's the last thread, it handles any remaining elements that couldn't be evenly distributed among threads.
 * A critical section (#pragma omp critical) is used to ensure that the updates to the total sum variable are performed atomically, avoiding race conditions.
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

        #pragma omp critical
        total += local_result;
    }

    return total;
}
