/*
 * Level 3: Advanced SIMD with SIMD Length Control (P3)
 * Description: Uses simdlen(8) to explicitly control SIMD vector length, optimizing for specific SIMD hardware configurations.
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                #pragma omp parallel for simd simdlen(8) private(j) aligned(src, dst: 32)
                for (int m = 0; m < flt_height; m++) {
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

