#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include "utils.h"

#define OPEN

void BFSGraph(int argc, char** argv);

struct Node {
    int starting;
    int no_of_edges;
};

void Usage(int argc, char** argv);
void readGraph(const char* input_file, int* no_of_nodes, struct Node** h_graph_nodes, bool** h_graph_mask, bool** h_updating_graph_mask, bool** h_graph_visited, int source, int* edge_list_size, int** h_graph_edges, int** h_cost);

void writeResult(const char* output_file, int no_of_nodes, int* h_cost);

#define NUM_TEAMS 1024
#define TEAM_SIZE 256

#ifdef __cplusplus
extern "C" {
#endif
extern void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size);
#ifdef __cplusplus
}
#endif
