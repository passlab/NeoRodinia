#include "bfs.h"

void Usage(int argc, char** argv) {
    fprintf(stderr, "Usage: %s <num_threads> <input_file>\n", argv[0]);
}

void readGraph(const char* input_file, int* no_of_nodes, struct Node** h_graph_nodes, bool** h_graph_mask, bool** h_updating_graph_mask, bool** h_graph_visited, int source, int* edge_list_size, int** h_graph_edges, int** h_cost) {
    
    // Read in Graph from a file
    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        fprintf(stderr, "Error Reading graph file\n");
        exit(EXIT_FAILURE);
    }
    fscanf(fp, "%d", no_of_nodes);
    // allocate host memory
    *h_graph_nodes = (struct Node*)malloc(sizeof(struct Node) * (*no_of_nodes));
    *h_graph_mask = (bool*)malloc(sizeof(bool) * (*no_of_nodes));
    *h_updating_graph_mask = (bool*)malloc(sizeof(bool) * (*no_of_nodes));
    *h_graph_visited = (bool*)malloc(sizeof(bool) * (*no_of_nodes));
    int start, edgeno;
    // initialize the memory
    for (unsigned int i = 0; i < *no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        (*h_graph_nodes)[i].starting = start;
        (*h_graph_nodes)[i].no_of_edges = edgeno;
        (*h_graph_mask)[i] = false;
        (*h_updating_graph_mask)[i] = false;
        (*h_graph_visited)[i] = false;
    }

    // read the source node from the file
    fscanf(fp, "%d", &source);
    //source = 0;

    // set the source node as true in the mask
    (*h_graph_mask)[source] = true;
    (*h_graph_visited)[source] = true;

    fscanf(fp, "%d", edge_list_size);

    int id, cost;
    *h_graph_edges = (int*)malloc(sizeof(int) * (*edge_list_size));
    for (int i = 0; i < *edge_list_size; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        (*h_graph_edges)[i] = id;
    }

    if (fp)
        fclose(fp);

    // allocate mem for the result on host side
    *h_cost = (int*)malloc(sizeof(int) * (*no_of_nodes));
    for (int i = 0; i < *no_of_nodes; i++) (*h_cost)[i] = -1;
    (*h_cost)[source] = 0;
}

void writeResult(const char* output_file, int no_of_nodes, int* h_cost) {
    // Store the result into a file
    FILE* fpo = fopen(output_file, "w");
    for (int i = 0; i < no_of_nodes; i++) fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
    fclose(fpo);
}
