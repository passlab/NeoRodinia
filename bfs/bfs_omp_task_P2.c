/*
 * Level 2: Taskloop with Grainsize (P2)
 * Description: Uses `taskloop` with `grainsize` to control the granularity of tasks, balancing workload by grouping nodes.
 */
#include "bfs.h"
#include <omp.h>

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size) {
    bool stop;

    do {
        stop = false;
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp taskloop grainsize(10)  // Group nodes into tasks with 10 nodes each
                for (int tid = 0; tid < no_of_nodes; tid++) {
                    if (h_graph_mask[tid]) {
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
