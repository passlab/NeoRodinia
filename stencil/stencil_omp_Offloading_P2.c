/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
 *
 */
#include "stencil.h"
#include <omp.h>

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int flt_size = flt_width*flt_height;
    int N = width*height;
    int i, j;

    #pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N])
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
