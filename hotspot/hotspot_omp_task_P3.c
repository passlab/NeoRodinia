/*
 * Level 3: Recursive Tasks
 * This version implements recursive task creation to process the grid hierarchically. The grid is divided into smaller sub-blocks, and tasks are created recursively until a base block size is reached. This approach is efficient for highly irregular workloads or when the grid size is large.
 *
 */
#include "hotspot.h"

// Recursive function to process grid blocks
void process_block(FLOAT *result, FLOAT *temp, FLOAT *power, int r_start, int r_end, int c_start, int c_end, int grid_cols, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, FLOAT step) {
    int block_rows = r_end - r_start;
    int block_cols = c_end - c_start;

    // Base case: process the block directly if its size is small enough
    if (block_rows <= BLOCK_SIZE_R && block_cols <= BLOCK_SIZE_C) {
        for (int r = r_start; r < r_end; ++r) {
            for (int c = c_start; c < c_end; ++c) {
                FLOAT delta;
                if ((r == 0 || r == grid_cols - 1) && (c == 0 || c == grid_cols - 1)) {
                    // Corner calculations
                    delta = (Cap_1) * (power[r * grid_cols + c] +
                        (temp[r * grid_cols + c + 1] - temp[r * grid_cols + c]) * Rx_1 +
                        (temp[(r + 1) * grid_cols + c] - temp[r * grid_cols + c]) * Ry_1 +
                        (amb_temp - temp[r * grid_cols + c]) * Rz_1);
                } else if (r == 0 || r == grid_cols - 1 || c == 0 || c == grid_cols - 1) {
                    // Edge calculations
                    delta = (Cap_1) * (power[r * grid_cols + c] +
                        ((temp[(r + 1) * grid_cols + c] + temp[(r - 1) * grid_cols + c] - 2.0 * temp[r * grid_cols + c]) * Ry_1) +
                        ((temp[r * grid_cols + c + 1] + temp[r * grid_cols + c - 1] - 2.0 * temp[r * grid_cols + c]) * Rx_1) +
                        (amb_temp - temp[r * grid_cols + c]) * Rz_1);
                } else {
                    // Regular calculations
                    delta = (Cap_1) * (power[r * grid_cols + c] +
                        ((temp[(r + 1) * grid_cols + c] + temp[(r - 1) * grid_cols + c] - 2.0 * temp[r * grid_cols + c]) * Ry_1) +
                        ((temp[r * grid_cols + c + 1] + temp[r * grid_cols + c - 1] - 2.0 * temp[r * grid_cols + c]) * Rx_1) +
                        (amb_temp - temp[r * grid_cols + c]) * Rz_1);
                }
                result[r * grid_cols + c] = temp[r * grid_cols + c] + delta;
            }
        }
        return;
    }

    // Recursive case: divide the block into four sub-blocks
    int r_mid = (r_start + r_end) / 2;
    int c_mid = (c_start + c_end) / 2;

    #pragma omp task shared(result, temp, power)
    process_block(result, temp, power, r_start, r_mid, c_start, c_mid, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);

    #pragma omp task shared(result, temp, power)
    process_block(result, temp, power, r_start, r_mid, c_mid, c_end, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);

    #pragma omp task shared(result, temp, power)
    process_block(result, temp, power, r_mid, r_end, c_start, c_mid, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);

    #pragma omp task shared(result, temp, power)
    process_block(result, temp, power, r_mid, r_end, c_mid, c_end, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);

    #pragma omp taskwait
}

void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int grid_rows, int grid_cols, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, FLOAT step) {
    FLOAT delta = 0;
    int r, c;
    int chunk;
    int num_chunk = grid_rows * grid_cols / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_grid_rows = grid_cols / BLOCK_SIZE_C;
    int chunks_in_grid_cols = grid_rows / BLOCK_SIZE_R;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Replace the outer parallel for loop with recursive task processing
            for (chunk = 0; chunk < num_chunk; ++chunk) {
                int r_start = BLOCK_SIZE_R * (chunk / chunks_in_grid_cols);
                int c_start = BLOCK_SIZE_C * (chunk % chunks_in_grid_rows);
                int r_end = r_start + BLOCK_SIZE_R > grid_rows ? grid_rows : r_start + BLOCK_SIZE_R;
                int c_end = c_start + BLOCK_SIZE_C > grid_cols ? grid_cols : c_start + BLOCK_SIZE_C;

                process_block(result, temp, power, r_start, r_end, c_start, c_end, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);
            }
        }
    }
}

void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int grid_rows, int grid_cols, int total_iterations) {
    FLOAT grid_height = chip_height / grid_rows;
    FLOAT grid_width = chip_width / grid_cols;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    if (need_full_report()) {
        fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", total_iterations, step);
        fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
    }

    FLOAT *r = result;
    FLOAT *t = temp;

    for (int i = 0; i < total_iterations; i++) {
        single_iteration(r, t, power, grid_rows, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);
        FLOAT *tmp = t;
        t = r;
        r = tmp;
    }
}
