/*
 * Level 2: Teams and Thread Hierarchies
 * Enhances Level 1 by introducing team-based parallelism with #pragma omp target teams distribute parallel for.
 * Divides the workload into teams, each consisting of threads, allowing better organization of GPU threads.
 * The hierarchical structure improves scalability and resource management for larger datasets.
 * Still uses default team and thread configurations but benefits from better load balancing compared to Level 1.
 */
#include "lud.h"
#include <omp.h>

void lud_kernel(float *a, int size) {
    int i, j, k;
    float sum;
    
    for (i = 0; i < size; i++) {
    #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum)
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++)
                sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }
        #pragma omp target teams distribute parallel for map(to:size) map(tofrom:a[0:size*size]) private(j, k, sum)
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++)
                sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

