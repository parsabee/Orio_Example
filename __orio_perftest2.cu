/*
TC:64
BC:14
SC:1
CB:True
PL:16
CFLAGS:
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#define N 1000
double *y;
double *x;
void malloc_arrays() {
  
  y = (double*) malloc((N) * sizeof(double));
  x = (double*) malloc((N) * sizeof(double));
}
void init_input_vars() {
  int i1;
  for (i1=0; i1<N; i1++)
   y[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   x[i1] = (i1) % 5 + 1;
}

double orio_t_start, orio_t_end, orio_t = (double)LONG_MAX;



__global__ void orcu_kernel352(const int n, double* y, double* x) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ double shared_y[64];
  __shared__ double shared_x[64];
  for (int i=tid; i<=n-1; i+=gsize) {
    shared_y[threadIdx.x]=y[i];
    shared_x[threadIdx.x]=x[i];
    shared_y[threadIdx.x]=shared_y[threadIdx.x]+shared_x[threadIdx.x];
    y[i]=shared_y[threadIdx.x];
  }
}


int main(int argc, char *argv[]) {
  
#ifdef MAIN_DECLARATIONS
  MAIN_DECLARATIONS()
#endif
  malloc_arrays();
  init_input_vars();

  cudaSetDeviceFlags(cudaDeviceBlockingSync);
  float orcu_elapsed=0.0, orcu_transfer=0.0;
  cudaEvent_t tstart, tstop, start, stop;
  cudaEventCreate(&tstart); cudaEventCreate(&tstop);
  cudaEventCreate(&start);  cudaEventCreate(&stop);
  
  for (int orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    
    
    

  int n=N;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, cacheBlocks=CB, preferL1Size=PL)

  for (i=0; i<=n-1; i++)
    y[i]+=x[i];

  ) @*/
  {
    cudaDeviceSynchronize();
    /*declare variables*/
    double* dev_y;
    double* dev_x;
    int nthreads=64;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=14;
    /*allocate device memory*/
    cudaMalloc(&dev_y,N*sizeof(double));
    cudaMalloc(&dev_x,N*sizeof(double));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    cudaMemcpy(dev_y,y,N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,N*sizeof(double),cudaMemcpyHostToDevice);
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel352<<<dimGrid,dimBlock>>>(n,dev_y,dev_x);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,N*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_x);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
  /*@ end @*/
  

    
    printf("{'[1, 0, 0, 0, 0, 0]' : (%g,%g)}\n", orcu_elapsed, orcu_transfer);
  }
  
  cudaEventDestroy(tstart); cudaEventDestroy(tstop);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  
  
  return 0;
}
