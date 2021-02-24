/*
TC:64
BC:28
SC:2
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



__global__ void orcu_kernel2260(const int n, double* y, double* x) {
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
    int nstreams=2;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=28;
    /*create streams*/
    int istream, soffset;
    cudaStream_t stream[nstreams+1];
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamCreate(&stream[istream]);
    int chunklen=n/nstreams;
    int chunkrem=n%nstreams;
    /*allocate device memory*/
    cudaMalloc(&dev_y,N*sizeof(double));
    cudaHostRegister(y,N*sizeof(double),cudaHostRegisterPortable);
    cudaMalloc(&dev_x,N*sizeof(double));
    cudaHostRegister(x,N*sizeof(double),cudaHostRegisterPortable);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_y+soffset,y+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_y+soffset,y+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    int blks4chunk=dimGrid.x/nstreams;
    if (dimGrid.x%nstreams!=0) 
      blks4chunk++ ;
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      orcu_kernel2260<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_y+soffset,dev_x+soffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      orcu_kernel2260<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_y+soffset,dev_x+soffset);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(y+soffset,dev_y+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(y+soffset,dev_y+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamSynchronize(stream[istream]);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_x);
    cudaHostUnregister(y);
    cudaHostUnregister(x);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
  /*@ end @*/
  

    
    printf("{'[1, 1, 1, 0, 0, 0]' : (%g,%g)}\n", orcu_elapsed, orcu_transfer);
  }
  
  cudaEventDestroy(tstart); cudaEventDestroy(tstop);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  
  
  return 0;
}
