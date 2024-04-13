#include "stencil.h"
#include "utils.h"

void stencil_serial(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j;
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

int main(int argc, char *argv[]) {
    
    int default_n = PROBLEM;
    int default_m = PROBLEM;

    int n = (argc > 1) ? atoi(argv[1]) : default_n;
    int m = (argc > 2) ? atoi(argv[2]) : default_m;

    REAL *input = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_serial = (REAL *) malloc(sizeof(REAL) * n * m);
    initialize(n, m, input);

    const float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0,  0, 1, 0, 0, },
        { 0,  0, 2, 0, 0, },
        { 3,  4, 5, 6, 7, },
        { 0,  0, 8, 0, 0, },
        { 0,  0, 9, 0, 0, },
    };

    int width = m;
    int height = n;
    
    stencil_serial(input, result_serial, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    
    double elapsed = read_timer_ms();
    stencil_kernel(input, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    elapsed = (read_timer_ms() - elapsed);
    
    double dif = 0;
    for (int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += fabs(result[i] - result_serial[i]);
    }
    
    if (need_full_report()) {
        printf("Problem Size(n,m): %d, %d\n", n,m);
        printf("Execution time(ms): %g\n", elapsed);
        if (need_verify()) printf("verify dif = %g\n", dif);
    } else {
        printf("%4f\n",elapsed);
    }

    free(input);
    free(result);
    return 0;
    
}
