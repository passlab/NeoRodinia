/*
 * Serial Version
 *
 */
#include "hotspot.h"

void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int grid_rows, int grid_cols, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, FLOAT step) {
    FLOAT delta=0;
    int r, c;
    int chunk;
    int num_chunk = grid_rows*grid_cols / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_grid_rows = grid_cols/BLOCK_SIZE_C;
    int chunks_in_grid_cols = grid_rows/BLOCK_SIZE_R;

    #pragma omp parallel for shared(power, temp, result) private(chunk, r, c, delta) firstprivate(grid_rows, grid_cols, num_chunk, chunks_in_grid_rows)
    for ( chunk = 0; chunk < num_chunk; ++chunk ) {
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_grid_cols);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_grid_rows);
        int r_end = r_start + BLOCK_SIZE_R > grid_rows ? grid_rows : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > grid_cols ? grid_cols : c_start + BLOCK_SIZE_C;
       
        if ( r_start == 0 || c_start == 0 || r_end == grid_rows || c_end == grid_cols ) {
            for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[grid_cols] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }    /* Corner 2 */
                    else if ((r == 0) && (c == grid_cols-1)) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c-1] - temp[c]) * Rx_1 +
                            (temp[c+grid_cols] - temp[c]) * Ry_1 +
                        (   amb_temp - temp[c]) * Rz_1);
                    }    /* Corner 3 */
                    else if ((r == grid_rows-1) && (c == grid_cols-1)) {
                        delta = (Cap_1) * (power[r*grid_cols+c] +
                            (temp[r*grid_cols+c-1] - temp[r*grid_cols+c]) * Rx_1 +
                            (temp[(r-1)*grid_cols+c] - temp[r*grid_cols+c]) * Ry_1 +
                        (   amb_temp - temp[r*grid_cols+c]) * Rz_1);
                    }    /* Corner 4    */
                    else if ((r == grid_rows-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*grid_cols] +
                            (temp[r*grid_cols+1] - temp[r*grid_cols]) * Rx_1 +
                            (temp[(r-1)*grid_cols] - temp[r*grid_cols]) * Ry_1 +
                            (amb_temp - temp[r*grid_cols]) * Rz_1);
                    }    /* Edge 1 */
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 +
                            (temp[grid_cols+c] - temp[c]) * Ry_1 +
                            (amb_temp - temp[c]) * Rz_1);
                    }    /* Edge 2 */
                    else if (c == grid_cols-1) {
                        delta = (Cap_1) * (power[r*grid_cols+c] +
                            (temp[(r+1)*grid_cols+c] + temp[(r-1)*grid_cols+c] - 2.0*temp[r*grid_cols+c]) * Ry_1 +
                            (temp[r*grid_cols+c-1] - temp[r*grid_cols+c]) * Rx_1 +
                            (amb_temp - temp[r*grid_cols+c]) * Rz_1);
                    }    /* Edge 3 */
                    else if (r == grid_rows-1) {
                        delta = (Cap_1) * (power[r*grid_cols+c] +
                            (temp[r*grid_cols+c+1] + temp[r*grid_cols+c-1] - 2.0*temp[r*grid_cols+c]) * Rx_1 +
                            (temp[(r-1)*grid_cols+c] - temp[r*grid_cols+c]) * Ry_1 +
                            (amb_temp - temp[r*grid_cols+c]) * Rz_1);
                    }    /* Edge 4 */
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*grid_cols] +
                            (temp[(r+1)*grid_cols] + temp[(r-1)*grid_cols] - 2.0*temp[r*grid_cols]) * Ry_1 +
                            (temp[r*grid_cols+1] - temp[r*grid_cols]) * Rx_1 +
                            (amb_temp - temp[r*grid_cols]) * Rz_1);
                    }
                    result[r*grid_cols+c] =temp[r*grid_cols+c]+ delta;
                }
            }
            continue;
        }

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
            for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
            /* Update Temperatures */
                result[r*grid_cols+c] =temp[r*grid_cols+c]+
                     ( Cap_1 * (power[r*grid_cols+c] +
                    (temp[(r+1)*grid_cols+c] + temp[(r-1)*grid_cols+c] - 2.f*temp[r*grid_cols+c]) * Ry_1 +
                    (temp[r*grid_cols+c+1] + temp[r*grid_cols+c-1] - 2.f*temp[r*grid_cols+c]) * Rx_1 +
                    (amb_temp - temp[r*grid_cols+c]) * Rz_1));
            }
        }
    }
}

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int grid_rows, int grid_cols, int total_iterations) {

    FLOAT grid_height = chip_height / grid_rows;
    FLOAT grid_width = chip_width / grid_cols;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1=1.f/Rx;
    FLOAT Ry_1=1.f/Ry;
    FLOAT Rz_1=1.f/Rz;
    FLOAT Cap_1 = step/Cap;
    if (need_full_report()) {
        fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", total_iterations, step);
        fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
    }
    FLOAT* r = result;
    FLOAT* t = temp;
    for (int i = 0; i < total_iterations ; i++) {
        single_iteration(r, t, power, grid_rows, grid_cols, Cap_1, Rx_1, Ry_1, Rz_1, step);
        FLOAT* tmp = t;
        t = r;
        r = tmp;
    }
}

