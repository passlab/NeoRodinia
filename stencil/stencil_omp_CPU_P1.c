/*
 * Level 1: Basic Parallel Execution (P1)
 * Description: Basic parallel execution using parallel for to distribute rows across threads.
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j, m, n;
    #pragma omp parallel for private(j, m, n)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            REAL sum = 0;
            for (n = 0; n < flt_width; n++) {
                for (m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m * flt_width + n;
                        sum += src[y * width + x] * filter[idx];
                    }
                }
            }
            dst[i * width + j] = sum;
        }
    }
}


