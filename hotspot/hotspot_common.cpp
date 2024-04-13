#include "hotspot.h"

void fatal(const char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == NULL) {
        printf("The file was not opened\n");
        return;
    }

    for (i = 0; i < grid_rows; i++) {
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }
    }

    fclose(fp);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
    int i;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen(file, "r");
    if (!fp)
        fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}
