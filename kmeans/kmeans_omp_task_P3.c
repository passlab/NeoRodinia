/*
 * Level 3: Recursive Tasks
 * Utilizes recursive tasks to divide the range of points dynamically. The recursion continues until a threshold is reached, at which point the tasks are executed. This allows for dynamic task generation and better load balancing for irregular workloads.
 *
 */
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

void process_points_recursive(int start, int end, float **feature, int nfeatures, int nclusters,
                              float **clusters, int *membership, int *new_centers_len,
                              float **new_centers, float *delta) {
    if (end - start <= 128) {  // Base case: small enough range
        for (int i = start; i < end; i++) {
            /* find the index of nearest cluster centers */
            int index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                #pragma omp atomic
                *delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers */
            #pragma omp atomic
            new_centers_len[index]++;

            for (int j = 0; j < nfeatures; j++) {
                #pragma omp atomic
                new_centers[index][j] += feature[i][j];
            }
        }
    } else {  // Recursive case: split the range
        int mid = (start + end) / 2;

        #pragma omp task shared(delta, feature, clusters, membership, new_centers_len, new_centers)
        process_points_recursive(start, mid, feature, nfeatures, nclusters, clusters,
                                 membership, new_centers_len, new_centers, delta);

        #pragma omp task shared(delta, feature, clusters, membership, new_centers_len, new_centers)
        process_points_recursive(mid, end, feature, nfeatures, nclusters, clusters,
                                 membership, new_centers_len, new_centers, delta);

        #pragma omp taskwait
    }
}

float **kmeans_clustering(float **feature, int nfeatures, int npoints, int nclusters,
                          float threshold, int *membership) {
    int i, j, n = 0;
    float delta;
    int *new_centers_len;    /* [nclusters]: no. of points in each cluster */
    float **clusters;        /* out: [nclusters][nfeatures] */
    float **new_centers;     /* [nclusters][nfeatures] */

    /* allocate space for cluster centers */
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick initial cluster centers */
    for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* allocate space for new centers and their lengths */
    new_centers_len = (int *)calloc(nclusters, sizeof(int));
    new_centers = (float **)malloc(nclusters * sizeof(float *));
    new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    do {
        delta = 0.0;

        #pragma omp parallel
        #pragma omp single
        {
            process_points_recursive(0, npoints, feature, nfeatures, nclusters,
                                     clusters, membership, new_centers_len,
                                     new_centers, &delta);
        }

        /* update cluster centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0;  /* reset to 0 */
            }
            new_centers_len[i] = 0;  /* reset to 0 */
        }
    } while (delta > threshold);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
