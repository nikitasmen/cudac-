
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


cudaError_t addWithCuda(int *c,  int *a,  int *b, unsigned long long size);

__global__ 
void addKernel(int *c,  int *a,  int *b , unsigned long long size )
{

    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if(i<size) c[i] = a[i] + b[i];
}

__host__
int main()
{
    clock_t start, end;
    double full_time; 
    
    long long  n = 200000000;
  
    int *a =(int*) malloc(sizeof(int) * n);
    int *b = (int*)malloc(sizeof(int) * n);
    int *c = (int*)malloc(sizeof(int) * n);

    if (c == NULL)
    {
        printf("Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < n ; ++i)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = 0;
    }
    

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, n);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
  
    
    for (int i = 0; i < n; ++i)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = 0;
    }
    start = clock(); 

    for (int i = 0; i < n; ++i)
    {
        c[i] = b[i] + a[i];
    }
    end = clock(); 
    full_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("serial : %lf", full_time);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c,  int *a,  int *b, unsigned long long size)
{
    clock_t start, end;
    double full_time;
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;


    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! c");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        system("pause");
        fprintf(stderr, "cudaMalloc failed! a");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! b");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! a");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! b ");
        goto Error;
    }
    // Launch a kernel on the GPU with one thread for each element.
    start = clock();
    addKernel<<<(long long) ceil((float)size/1024), 1024 >>>(dev_c, dev_a, dev_b, size);
    end = clock();
    full_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("parallel : %lf", full_time);
    


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
