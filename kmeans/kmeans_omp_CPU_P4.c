#include "kmeans.h"

int cluster(int numObjects, int numAttributes, float **attributes, int nclusters, float threshold, float ***cluster_centres) {
    int *membership;
    float **tmp_cluster_centres;
    membership = (int*) malloc(numObjects * sizeof(int));
    srand(7);
    tmp_cluster_centres = kmeans_clustering(attributes, numAttributes, numObjects, nclusters, threshold, membership);
    if (*cluster_centres) {
        free((*cluster_centres)[0]);
        free(*cluster_centres);
    }
    *cluster_centres = tmp_cluster_centres;
    free(membership);

    return 0;
}

int find_nearest_point(float *pt, int nfeatures, float **pts, int npts) {
    //*pt:[nfeatures]
    //**pts:[npts][nfeatures]
    int index, i;
    float min_dist=FLT_MAX;
    //find the cluster center id with min distance to pt
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); //no need square root
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

// multi-dimensional spatial Euclid distance square
float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}

float** kmeans_clustering(float** feature, int nfeatures, int npoints, int nclusters, float threshold, int* membership) {
    int i, j, k, n = 0, index, loop = 0;
    int* new_centers_len;      /* [nclusters]: no. of points in each cluster */
    float** new_centers;       /* [nclusters][nfeatures] */
    float** clusters;          /* out: [nclusters][nfeatures] */
    float delta;
    int nthreads = 128;
    int** partial_new_centers_len;
    float*** partial_new_centers;

    /* allocate space for returning variable clusters[] */
    clusters = (float**)malloc(nclusters * sizeof(float*));
    clusters[0] = (float*)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*)calloc(nclusters, sizeof(int));

    new_centers = (float**)malloc(nclusters * sizeof(float*));
    new_centers[0] = (float*)calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    partial_new_centers_len = (int**)malloc(nthreads * sizeof(int*));
    partial_new_centers_len[0] = (int*)calloc(nthreads * nclusters, sizeof(int));
    for (i = 1; i < nthreads; i++)
        partial_new_centers_len[i] = partial_new_centers_len[i - 1] + nclusters;

    partial_new_centers = (float***)malloc(nthreads * sizeof(float**));
    partial_new_centers[0] = (float**)malloc(nthreads * nclusters * sizeof(float*));
    for (i = 1; i < nthreads; i++)
        partial_new_centers[i] = partial_new_centers[i - 1] + nclusters;

    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < nclusters; j++)
            partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
    }

    do {
        delta = 0.0;

        omp_set_num_threads(nthreads);
        #pragma omp parallel for private(i,j,index) firstprivate(npoints,nclusters,nfeatures) shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len) schedule(dynamic, 128) reduction(+:delta)
        for (i = 0; i < npoints; i++) {
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            if (membership[i] != index)
                delta += 1.0;
            membership[i] = index;
            partial_new_centers_len[omp_get_thread_num()][index]++;
            for (j = 0; j < nfeatures; j++)
                partial_new_centers[omp_get_thread_num()][index][j] += feature[i][j];
        }

        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0;
                for (k = 0; k < nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0;
                }
            }
        }

        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;
            }
            new_centers_len[i] = 0;
        }
    } while (delta > threshold && loop++ < 500);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
