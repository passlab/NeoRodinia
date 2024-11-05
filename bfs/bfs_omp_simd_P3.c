/*
 * Level 3: Advanced SIMD with Explicit SIMD Length (P3)
 * Description: Specifies the SIMD vector length with `simdlen` for finer control over SIMD vectorization.
 */
#include "bfs.h"
#include <omp.h>

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {
    bool stop;

    do {
        stop = false;
        #pragma omp parallel for
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (h_graph_mask[tid]) {
                h_graph_mask[tid] = false;
                #pragma omp simd simdlen(8) aligned(h_graph_edges, h_cost: 32)
                for (int i = h_graph_nodes[tid].starting; i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
                    int id = h_graph_edges[i];
                    if (!h_graph_visited[id]) {
                        h_cost[id] = h_cost[tid] + 1;
                        h_updating_graph_mask[id] = true;
                    }
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
