/*
 * Level 3: Recursive Tasking (P3)
 * Description: Implements recursive tasking by splitting the stencil computation into smaller subtasks that handle sections of the input grid.
 */
#include "stencil.h"
#include <omp.h>

void stencil_task(int i_start, int i_end, int width, const REAL* src, REAL* dst, const float* filter, int flt_width, int flt_height) {
    if (i_end - i_start <= 64) {  // Base case
        for (int i = i_start; i < i_end; i++) {
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
    } else {
        int mid = (i_start + i_end) / 2;
        #pragma omp task
        stencil_task(i_start, mid, width, src, dst, filter, flt_width, flt_height);
        #pragma omp task
        stencil_task(mid, i_end, width, src, dst, filter, flt_width, flt_height);
        #pragma omp taskwait
    }
}

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    #pragma omp parallel
    {
        #pragma omp single
        stencil_task(0, height, width, src, dst, filter, flt_width, flt_height);
    }
}
