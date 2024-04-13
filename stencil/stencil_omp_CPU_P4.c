/*
 * Compared to P3, in P4, the `simd` directive is used within the innermost loop to indicate that the loop can be vectorized using SIMD (Single Instruction, Multiple Data) instructions for further optimization.
 *
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j;
    #pragma omp parallel for schedule(dynamic, 64) private(i,j) collapse(2)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            REAL sum = 0;
            int m, n;
            for (n = 0; n < flt_width; n++) {
                #pragma omp simd
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

