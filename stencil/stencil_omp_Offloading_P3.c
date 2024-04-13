/*
 * Baed on P2, P3 implements collapse, num_teams and num_threads clauses.
 * The `collapse(2)` clause combines the nested loops into a single loop, effectively parallelizing both the `i` and `j` iterations. This can improve parallelization efficiency by reducing loop overhead.
 * The `num_teams(NUM_TEAMS)` and `num_threads(TEAM_SIZE)` clauses specify the number of teams and the number of threads per team, respectively.
 * These parameters allow fine-grained control over the parallel execution, optimizing resource utilization.
 *
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int flt_size = flt_width*flt_height;
    int N = width*height;
    int i, j;

    #pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(2) num_teams(NUM_TEAMS) num_threads(TEAM_SIZE)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            REAL sum = 0;
            int m, n;
            for (n = 0; n < flt_width; n++) {
                for (m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}
