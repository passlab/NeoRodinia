/*
 * Level 1: Basic GPU Offloading
 * This version utilizes OpenMP target parallel to offload the main computational loops to the GPU. All feature points, membership arrays, and cluster centers are mapped to the device. Each GPU thread computes the distance between a point and all cluster centers to determine the closest cluster. The algorithm performs reduction manually on the host to update cluster centers.
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


#pragma omp declare target
int find_nearest_point(float *pt, int nfeatures, float **pts, int npts) {
    int index, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++) {
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
    }

    return ans;
}
#pragma omp end declare target

float **kmeans_clustering(float **h_feature, int nfeatures, int npoints, int nclusters, float threshold, int *membership) {
    int i, j, k, n = 0, index, loop = 0;
    int *new_centers_len;  /* [nclusters]: no. of points in each cluster */
    float **h_new_centers; /* [nclusters][nfeatures] */
    float **h_clusters;    /* out: [nclusters][nfeatures] */
    float delta;

    int nthreads;
    int **h_partial_new_centers_len;
    float **h_partial_new_centers;

    int num_omp_threads = 128;
    nthreads = num_omp_threads;

    int host_id = omp_get_initial_device();
    int device_id = omp_get_default_device();

    float **feature = malloc(npoints * sizeof(float *));
    for (i = 0; i < npoints; i++) {
        feature[i] = omp_target_alloc(nfeatures * sizeof(float), device_id);
        omp_target_memcpy(feature[i], h_feature[i], nfeatures * sizeof(float), 0, 0,
                          device_id, host_id);
    }

    /* allocate space for returning variable clusters[] */
    h_clusters = (float **)malloc(nclusters * sizeof(float *));
    h_clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        h_clusters[i] = h_clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int)rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            h_clusters[i][j] = h_feature[n][j];
        n++;
    }

    float **clusters = malloc(nclusters * sizeof(float *));
    for (i = 0; i < nclusters; i++) {
        clusters[i] = omp_target_alloc(nfeatures * sizeof(float), device_id);
        omp_target_memcpy(clusters[i], h_clusters[i], nfeatures * sizeof(float), 0,
                          0, device_id, host_id);
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *)calloc(nclusters, sizeof(int));

    h_new_centers = (float **)malloc(nclusters * sizeof(float *));
    h_new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        h_new_centers[i] = h_new_centers[i - 1] + nfeatures;

    float **new_centers = malloc(nclusters * sizeof(float *));
    for (i = 0; i < nclusters; i++) {
        new_centers[i] = omp_target_alloc(nfeatures * sizeof(float), device_id);
        omp_target_memcpy(new_centers[i], h_new_centers[i],
                          nfeatures * sizeof(float), 0, 0, device_id, host_id);
    }

    h_partial_new_centers_len = (int **)malloc(nthreads * sizeof(int *));
    h_partial_new_centers_len[0] =
        (int *)calloc(nthreads * nclusters, sizeof(int));
    for (i = 1; i < nthreads; i++)
        h_partial_new_centers_len[i] = h_partial_new_centers_len[i - 1] + nclusters;

    float **partial_new_centers_len = malloc(nthreads * sizeof(int *));
    for (i = 0; i < nthreads; i++) {
        partial_new_centers_len[i] =
            omp_target_alloc(nclusters * sizeof(int), device_id);
        omp_target_memcpy(partial_new_centers_len[i], h_partial_new_centers_len[i],
                          nclusters * sizeof(int), 0, 0, device_id, host_id);
    }

    h_partial_new_centers =
        (float **)malloc(nthreads * nclusters * sizeof(float *));
    for (i = 1; i < nthreads; i++)
        h_partial_new_centers[i] = h_partial_new_centers[i - 1] + nclusters;

    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < nclusters; j++)
            h_partial_new_centers[i * nclusters + j] =
                (float *)calloc(nfeatures, sizeof(float));
    }
    float **partial_new_centers = malloc(nthreads * nclusters * sizeof(float *));
    for (i = 0; i < nthreads; i++) {
        for (j = 0; j < nclusters; j++) {
            partial_new_centers[i * nclusters + j] =
                omp_target_alloc(nfeatures * sizeof(float), device_id);
            omp_target_memcpy(partial_new_centers[i * nclusters + j],
                              h_partial_new_centers[i * nclusters + j],
                              nfeatures * sizeof(float), 0, 0, device_id, host_id);
        }
    }
    int total_num_threads = 0;
    if (npoints <= num_omp_threads) {
        total_num_threads = npoints;
    } else {
        total_num_threads = num_omp_threads;
    }
#pragma omp target data map(to                                                 \
                            : feature [0:npoints], membership [0:npoints],     \
                              new_centers [0:nclusters],                       \
                              new_centers_len [0:nclusters])                   \
    map(to                                                                     \
        : clusters [0:nclusters],                                              \
          partial_new_centers [0:nthreads * nclusters],                        \
          partial_new_centers_len [0:nthreads])
    {
        do {
            delta = 0.0;
#pragma omp target parallel map(tofrom : delta) num_threads(num_omp_threads)
#pragma omp for private(i, j, index)                                           \
    firstprivate(npoints, nclusters, nfeatures) reduction(+ : delta)
            for (i = 0; i < npoints; i++) {
                int tid = i % total_num_threads;
                /* find the index of nestest cluster centers */
                index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index)
                    delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of all objects located
                       within */
                partial_new_centers_len[tid][index]++;
                for (j = 0; j < nfeatures; j++)
                    partial_new_centers[tid * nclusters + index][j] += feature[i][j];
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
#pragma omp target parallel for private(k)
            for (i = 0; i < nclusters; i++) {
                for (j = 0; j < nthreads; j++) {
                    new_centers_len[i] += partial_new_centers_len[j][i];
                    partial_new_centers_len[j][i] = 0.0;
                    for (k = 0; k < nfeatures; k++) {
                        new_centers[i][k] += partial_new_centers[j * nclusters + i][k];
                        partial_new_centers[j * nclusters + i][k] = 0.0;
                    }
                }
            }

            /* replace old cluster centers with new_centers */
#pragma omp target parallel for
            for (i = 0; i < nclusters; i++) {
                for (j = 0; j < nfeatures; j++) {
                    if (new_centers_len[i] > 0)
                        clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                    new_centers[i][j] = 0.0; /* set back to 0 */
                }
                new_centers_len[i] = 0; /* set back to 0 */
            }
        } while (delta > threshold && loop++ < 500);
    }

    for (i = 0; i < nclusters; i++) {
        omp_target_memcpy(h_clusters[i], clusters[i], nfeatures * sizeof(float), 0,
                          0, host_id, device_id);
        omp_target_free(clusters[i], device_id);
    }

    for (i = 0; i < npoints; i++) {
        omp_target_free(feature[i], device_id);
    }

    free(h_new_centers[0]);
    free(h_new_centers);
    free(new_centers_len);

    return h_clusters;
}
