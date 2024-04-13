/*
 * This kernel implements a Breadth-First Search (BFS) algorithm using CUDA for parallel processing on GPU.
 *
 */

#include "bfs.h"
#define MAX_THREADS_PER_BLOCK 512

#ifndef _KERNEL_H_
#define _KERNEL_H_
//This kernel processes the nodes in the graph. For each node, it checks its neighboring nodes and updates their cost if they haven't been visited yet.
__global__ void Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) {
    int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if( tid<no_of_nodes && g_graph_mask[tid]) {
        g_graph_mask[tid]=false;
        for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++) {
            int id = g_graph_edges[i];
            if(!g_graph_visited[id]) {
                g_cost[id]=g_cost[tid]+1;
                g_updating_graph_mask[id]=true;
            }
        }
    }
}
#endif 

#ifndef _KERNEL2_H_
#define _KERNEL2_H_
//This kernel updates the graph mask and visited status based on the nodes that were updated in the previous step and signals if the BFS traversal is over.
__global__ void Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes) {
    int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if( tid<no_of_nodes && g_updating_graph_mask[tid]) {
        g_graph_mask[tid]=true;
        g_graph_visited[tid]=true;
        *g_over=true;
        g_updating_graph_mask[tid]=false;
    }
}

#endif

void bfs_kernel(int no_of_nodes, struct Node* h_graph_nodes, bool* h_graph_mask, bool* h_updating_graph_mask, bool* h_graph_visited, int* h_graph_edges, int* h_cost, int edge_list_size){
    //copy the Node list to device memory
    struct Node* d_graph_nodes;
    cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
    cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
    //Copy the Edge List to device Memory
    int* d_graph_edges;
    cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
    cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

    //Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
    cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

    bool* d_updating_graph_mask;
    cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
    cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
    cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
    // allocate device memory for result
    int* d_cost;
    cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
    cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

    //make a bool to check if the execution is over
    bool *d_over;
    cudaMalloc( (void**) &d_over, sizeof(bool));

    int num_of_blocks = 1;
    int num_of_threads_per_block = no_of_nodes;

    //Make execution Parameters according to the number of nodes
    //Distribute threads across multiple Blocks if necessary
    if(no_of_nodes>MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
    }
	
    // setup execution parameters
    dim3  grid( num_of_blocks, 1, 1);
    dim3  threads( num_of_threads_per_block, 1, 1);
       
    int k=0;
    printf("Start traversing the tree\n");
    bool stop;
	
    do {
        //if no thread changes this value then the loop stops
        stop=false;
        cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
        Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);		
        Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);	

        cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
        k++;
    } while(stop);
    printf("Kernel Executed %d times\n",k);

    // copy result from device to host
    cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;	
}