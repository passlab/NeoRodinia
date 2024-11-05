/*
 * Level 3: Recursive Task Creation (P3)
 * Description: Uses recursive tasks to process each node and its neighbors hierarchically, allowing finer control over task creation.
 */
#include "bfs.h"
#include <omp.h>

void bfs_recursive(int tid, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost) {
    if (!h_graph_mask[tid]) return;

    h_graph_mask[tid] = false;
    for (int i = h_graph_nodes[tid].starting; i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
        int id = h_graph_edges[i];
        if (!h_graph_visited[id]) {
            h_cost[id] = h_cost[tid] + 1;
            h_updating_graph_mask[id] = true;
        }
    }

    #pragma omp taskwait
}

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {
    bool stop;

    do {
        stop = false;

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int tid = 0; tid < no_of_nodes; tid++) {
                    #pragma omp task
                    bfs_recursive(tid, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost);
                }
            }
        }

        #pragma omp parallel for
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_updating_graph_mask[tid]) {
                h_graph_mask[tid] = true;
                h_graph_visited[tid] = true;
                stop = true;
                h_updating_graph_mask[tid] = false;
            }
        }
    } while (stop);
}
