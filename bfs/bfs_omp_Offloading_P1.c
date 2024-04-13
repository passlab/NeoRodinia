/*
 * The computation is parallelized using OpenMP's target parallel for directive, distributing the workload across available threads.
 * Memory mapping directives ensure proper data sharing and synchronization between threads.
 *
 */
#include "bfs.h"
#include <omp.h>

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {
    bool stop;
#pragma omp target data map(to : no_of_nodes, h_graph_mask [0:no_of_nodes],    \
      h_graph_nodes [0:no_of_nodes], h_graph_edges [0:edge_list_size],         \
      h_graph_visited [0:no_of_nodes], h_updating_graph_mask [0:no_of_nodes])  \
    map(tofrom : h_cost [0:no_of_nodes])
    do {
        // if no thread changes this value then the loop stops
        stop = false;
        #pragma omp target parallel for
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_graph_mask[tid] == true) {
                h_graph_mask[tid] = false;
                for (int i = h_graph_nodes[tid].starting; i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
                    int id = h_graph_edges[i];
                    if (!h_graph_visited[id]) {
                        h_cost[id] = h_cost[tid] + 1;
                        h_updating_graph_mask[id] = true;
                    }
                }
            }
        }
        #pragma omp target parallel for map(tofrom : stop)
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_updating_graph_mask[tid] == true) {
                h_graph_mask[tid] = true;
                h_graph_visited[tid] = true;
                stop = true;
                h_updating_graph_mask[tid] = false;
            }
        }
    } while (stop);
}
