/*
 * Compared to P3, the `num_teams(size/TEAM_SIZE)` and `num_threads(TEAM_SIZE)` clauses in P4 dynamically adjust the number of teams and threads based on the size of the input matrices, achieving one element per thread.
 *
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int flt_size = flt_width*flt_height;
    int N = width*height;
    int i, j;

    #pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(2) num_teams(N/TEAM_SIZE) num_threads(TEAM_SIZE)
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
