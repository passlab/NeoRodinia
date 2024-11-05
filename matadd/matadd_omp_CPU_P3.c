/*
 * Level 3: Tiling with Nested Parallelism (P3)
 * Description: Uses tiling to break the matrix into blocks for better cache usage. Inner parallelism is applied at the tile level.
 */
#include "matadd.h"
#include <omp.h>

void matadd_kernel(int N, REAL *C, REAL *A) {
    int tile_size = 32;  // Tile size can be adjusted based on cache size and matrix dimensions
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            // Compute tile addition
            for (int ti = i; ti < i + tile_size && ti < N; ++ti) {
                for (int tj = j; tj < j + tile_size && tj < N; ++tj) {
                    C[ti * N + tj] += A[ti * N + tj];
                }
            }
        }
    }
}
