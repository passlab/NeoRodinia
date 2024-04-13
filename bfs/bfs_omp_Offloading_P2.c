/*
 * Compared with P1, P2 implements the `target teams distribute parallel for` directive.
 * It first distributes the loop iterations among teams of threads, and then each team further distributes the iterations among threads within the team.
 * This directive provides more flexibility in how work is divided among threads, allowing for potentially better load balancing and resource utilization.
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
        #pragma omp target teams distribute parallel for
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
        #pragma omp target teams distribute parallel for map(tofrom : stop)
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
