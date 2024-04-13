/***********************************************
        streamcluster_omp.cpp
        : parallelized code of streamcluster using OpenMP

        - original code from PARSEC Benchmark Suite
        - parallelization with OpenMP API has been applied by

        Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
        University of Virginia
        Department of Electrical and Computer Engineering
        Department of Computer Science

***********************************************/

#include <assert.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

using namespace std;

#define MAXNAMESIZE 1024 // max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 // number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 3 // iterate ITER* k log k times; ITER >= 1

// #define PRINTINFO //comment this out to disable output
#define PROFILE // comment this out to disable instrumentation code
// #define INSERT_WASTE //uncomment this to insert waste computation into dist
// function

#define CACHE_LINE 512 // cache line in byte

/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
    float weight;
    float *coord;
    long assign; /* number of point where this one is assigned */
    float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
    long num; /* number of points; may not be N if this is a sample */
    int dim;  /* dimensionality */
    Point *p; /* the array itself */
} Points;

class PStream {
  public:
    virtual size_t read(float *dest, int dim, int num) = 0;
    virtual int ferror() = 0;
    virtual int feof() = 0;
    virtual ~PStream() {}
};

// synthetic stream
class SimStream : public PStream {
  public:
    SimStream(long n_) { n = n_; }
    size_t read(float *dest, int dim, int num) {
        size_t count = 0;
        for (int i = 0; i < num && n > 0; i++) {
            for (int k = 0; k < dim; k++) {
                dest[i * dim + k] = lrand48() / (float)INT_MAX;
            }
            n--;
            count++;
        }
        return count;
    }
    int ferror() { return 0; }
    int feof() { return n <= 0; }
    ~SimStream() {}

  private:
    long n;
};

class FileStream : public PStream {
  public:
    FileStream(char *filename) {
        fp = fopen(filename, "rb");
        if (fp == NULL) {
            fprintf(stderr, "error opening file %s\n.", filename);
            exit(1);
        }
    }
    size_t read(float *dest, int dim, int num) {
        return std::fread(dest, sizeof(float) * dim, num, fp);
    }
    int ferror() { return std::ferror(fp); }
    int feof() { return std::feof(fp); }
    ~FileStream() {
        printf("closing file stream\n");
        fclose(fp);
    }

  private:
    FILE *fp;
};

float dist(Point p1, Point p2, int dim);
void streamCluster(PStream *stream, long kmin, long kmax, int dim,
                   long chunksize, long centersize, char *outfile);
double pgain_kernel(long x, Points *points, double z, long int *numcenters,
                    int pid);
double streamCluster_wrapper(PStream *stream, long kmin, long kmax, int dim,
                             long chunksize, long centersize, char *outfile);

#define NUM_TEAMS 1024
#define TEAM_SIZE 512
