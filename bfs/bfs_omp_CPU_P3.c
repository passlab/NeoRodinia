/*
 * Compared to P2, P3 implements the `collapse(2)` clause, combining the nested loops into a single loop, effectively parallelizing both the `tid` and `i` iterations. This can improve parallelization efficiency by reducing loop overhead.
 *
 */
#include "bfs.h"
#include <omp.h>

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {

    bool stop;

    do {
        // if no thread changes this value then the loop stops
        stop = false;
        #pragma omp parallel for shared(h_graph_mask, h_graph_nodes) schedule(dynamic, 128) collapse(2)
        for (int tid = 0; tid < no_of_nodes; tid++) {
            for (int i = h_graph_nodes[tid].starting; i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
                if (h_graph_mask[tid] == true) {
                    h_graph_mask[tid] = false;
                    int id = h_graph_edges[i];
                    if (!h_graph_visited[id]) {
                        h_cost[id] = h_cost[tid] + 1;
                        h_updating_graph_mask[id] = true;
                    }
                }
            }
        }
        #pragma omp parallel shared(h_graph_mask, h_graph_visited)
        #pragma omp for schedule(static, 128)
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
