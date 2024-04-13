#include "hotspot.h"
#include "utils.h"

int main(int argc, char **argv) {
    int grid_rows = 0, grid_cols = 0, sim_time = 0;
    int total_iterations = 60;
    FLOAT *temp, *power, *result;
    char *tfile, *pfile, *ofile;

    /* check validity of inputs */
    if (argc != 7)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 ||
        (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0)
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays */
    temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power)
        fatal("unable to allocate memory");

    /* read initial temperatures and input power */
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    if (need_full_report()) printf("Start computing the transient temperature\n");

    double elapsed = read_timer_ms();

    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols, total_iterations);
    
    elapsed = (read_timer_ms() - elapsed);

    if (need_full_report()) {
        printf("Ending simulation\n");
        printf("Total time: %4f\n", elapsed);
    } else {
        printf("%4f\n", elapsed);
    }

    writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);

    /* cleanup */
    free(temp);
    free(power);

    return 0;
}
