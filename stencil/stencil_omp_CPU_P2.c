/*
 * P2 introduces guided scheduling.
 * The code achieves parallel execution with dynamic adjustment of the chunk size based on runtime behavior.
 * This can lead to better load balancing and improved performance compared to static scheduling, especially for cases where workload distribution across iterations varies significantly.
 *
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j;
    #pragma omp parallel for schedule(dynamic, 64) private(i,j)
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

