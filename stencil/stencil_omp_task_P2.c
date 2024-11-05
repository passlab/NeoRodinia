/*
 * Level 2: Taskloop with Specified Grain Size (P2)
 * Description: Introduces a grainsize clause to control task granularity, allowing the outer loop to process blocks of rows.
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(8)  // Specify grainsize for balanced workload
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    REAL sum = 0;
                    for (int n = 0; n < flt_width; n++) {
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
    }
}
