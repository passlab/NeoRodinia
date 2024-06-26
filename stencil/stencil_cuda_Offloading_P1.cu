/*
 * Each thread computes one pixel, the whole image is in global memory, filter is in global memory as well. filter size is parametered. 
 * For fixed filter size, we can put the filter in either register (as passing argument of the kernel) or shared memory
 *
 */
#include "stencil.h"
#define BLOCK_SIZE 16

__global__ void global_element(REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    REAL sum = 0;
    for (int n = 0; n < flt_width; n++) {
        for (int m = 0; m < flt_height; m++) {
            int x = j + n - flt_width / 2;
            int y = i + m - flt_height / 2;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                int idx = m * flt_width + n;
                sum += src[y * width + x] * filter[idx];
            }
        }
    }

    // Each thread writes one element to C matrix
    dst[i*width + j] = sum;
}

void stencil_kernel(const REAL* src, REAL* dst, int width, int height, const float* filter, int fil_width, int fil_height) {
    REAL *input_device, *output_device;
    float *filter_device;
    cudaMalloc(&input_device, width*height*sizeof(REAL));
    cudaMalloc(&output_device, width*height*sizeof(REAL));
    cudaMalloc(&filter_device, fil_width*fil_height*sizeof(float));

    cudaMemcpy(input_device, src, width*height*sizeof(REAL), cudaMemcpyHostToDevice);
    //cudaMemcpy(output_device, dst, width*height*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_device, filter, fil_width*fil_height*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    global_element<<<dimGrid, dimBlock>>>(input_device, output_device, width, height, filter_device, fil_width, fil_height);

    cudaMemcpy(dst, output_device, width*height*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(input_device);
    cudaFree(output_device);
    cudaFree(filter_device);
}
