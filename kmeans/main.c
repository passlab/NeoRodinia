#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename         : file containing data to be clustered\n"
        "       -b                  : input file is in binary format\n"
        "       -k                  : number of clusters (default is 5) \n"
        "       -t threshold        : threshold value\n"
    ;
    fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv) {
    int opt;
    extern char *optarg;
    extern int optind;
    int nclusters = 5;
    char *filename = 0;
    float *buf;
    float **attributes;
    float **cluster_centres = NULL;
    int i, j;
    int numAttributes;
    int numObjects;
    char line[1024];
    int isBinaryFile = 0;
    int nloops = 1;
    float threshold = 0.001;

    while ((opt = getopt(argc, argv, "i:k:t:b:?")) != EOF) {
        switch (opt) {
            case 'i':
                filename = optarg;
                break;
            case 'b':
                isBinaryFile = 1;
                break;
            case 't':
                threshold = atof(optarg);
                break;
            case 'k':
                nclusters = atoi(optarg);
                break;
            case '?':
                usage(argv[0]);
                break;
            default:
                usage(argv[0]);
                break;
        }
    }

    if (filename == 0)
        usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/

    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects, sizeof(int));
        read(infile, &numAttributes, sizeof(int));

        /* allocate space for attributes[] and read attributes of all objects */
        buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
        attributes = (float **) malloc(numObjects * sizeof(float *));
        attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
        for (i = 1; i < numObjects; i++) {
            attributes[i] = attributes[i - 1] + numAttributes;
        }
        read(infile, buf, numObjects * numAttributes * sizeof(float));

        close(infile);
    } else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
            }
        }

        /* allocate space for attributes[] and read attributes of all objects */
        buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
        attributes = (float **) malloc(numObjects * sizeof(float *));
        attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
        for (i = 1; i < numObjects; i++) {
            attributes[i] = attributes[i - 1] + numAttributes;
        }
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j = 0; j < numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }
    if (need_full_report()) printf("I/O completed\n");

    memcpy(attributes[0], buf, numObjects * numAttributes * sizeof(float));
    
    double elapsed = read_timer();
    for (i = 0; i < nloops; i++) {
        cluster_centres = NULL;
        cluster(numObjects, numAttributes, attributes, nclusters, threshold, &cluster_centres);
    }
    elapsed = (read_timer() - elapsed);
    if (need_full_report()) {
        printf("Number of Clusters: %d\n", nclusters);
        printf("Number of Attributes: %d\n\n", numAttributes);
        printf("Compute time: %f\n", elapsed);
        
        if (need_verify()) {
            printf("\nCluster Centers Output\n");
            printf("The first number is the cluster number and the following data are attribute values\n");
            printf("================================================================================\n\n");

            for (i = 0; i < nclusters; i++) {
                printf("%d: ", i);
                for (j = 0; j < numAttributes; j++) {
                    printf("%.2f ", cluster_centres[i][j]);
                }
                printf("\n\n");
            }
        }
    } else {
        printf("%f\n", elapsed);
    }
    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return (0);
}
