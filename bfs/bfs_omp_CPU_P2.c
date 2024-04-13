/*
 * This kernel utilizes dynamic scheduling with a chunk size of 128 for load balancing across threads.
 * With the `schedule(dynamic, 128)` clause, tasks are dynamically assigned to threads, with each thread processing a chunk of 128 iterations at a time. This dynamic scheduling helps to balance the workload more efficiently, especially when the iteration times vary significantly.
 *
 */
#include "bfs.h"
#include <omp.h>

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {

    bool stop;

    do {
        // if no thread changes this value then the loop stops
        stop = false;
        #pragma omp parallel shared(h_graph_mask, h_graph_nodes)
        #pragma omp for schedule(dynamic, 128)
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
        #pragma omp parallel shared(h_graph_mask, h_graph_visited)
        #pragma omp for schedule(dynamic, 128)
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
