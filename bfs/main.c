#include "bfs.h"
#include "utils.h"

int main(int argc, char** argv) {
    
    char* default_input_file = "../data/bfs/graph1MW_6.txt";
    
    char* input_file = (argc > 1) ? strdup(argv[1]) : strdup(default_input_file);
    
    // Read graph data
    int no_of_nodes, edge_list_size;
    int source = 0;
    struct Node* h_graph_nodes;
    bool* h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
    int* h_graph_edges, *h_cost;
    readGraph(input_file, &no_of_nodes, &h_graph_nodes, &h_graph_mask, &h_updating_graph_mask, &h_graph_visited, source, &edge_list_size, &h_graph_edges, &h_cost);
    
    double elapsed_omp_parallel_for = read_timer_ms();
    bfs_kernel(no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost, edge_list_size);
    elapsed_omp_parallel_for = (read_timer_ms() - elapsed_omp_parallel_for);
    
    writeResult("result.txt", no_of_nodes, h_cost);

    if (need_full_report()) {
        printf("Result stored in result.txt\n");
        printf("======================================================================================================\n");
        printf("\tBreadth First Search\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Input file:: %s\n", input_file);
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Compute time: %4f\n", elapsed_omp_parallel_for);
        
    } else {
         printf("%4f\n",elapsed_omp_parallel_for);
    }
    

    
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    free(input_file);

    return 0;
}
