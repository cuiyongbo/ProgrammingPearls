// taken from `cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu`

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

__global__ void vector_add(const float* A, const float* B, float* C, int num_elements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num_elements) {
    C[i] = A[i] + B[i];
  }
}

#define err_quit(...) do {	\
	fprintf(stderr, "%s:%d %s: ", __FILE__, __LINE__, __FUNCTION__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, "\n");\
	exit(EXIT_FAILURE); } while(0)


int main() {
  /*
    CUDA programm ABC:
      1. prepare host and device memory
      2. h2d: copy vars from host memory to device memory
      3. perform computation on GPU
      4. d2h: copy computation result from device memory to host memory
      5. destroy device and host memory
  */
  
  // 1. prepare host and device memory
  int num_elements = 5000;
  int mem_size = num_elements * sizeof(float);
  float *h_A = (float*)malloc(mem_size);
  float *h_B = (float*)malloc(mem_size);
  float *h_C = (float*)malloc(mem_size);
  for (int i=0; i<num_elements; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  float* d_A = NULL;
  float* d_B = NULL;
  float* d_C = NULL;

/*
template<class T>
static __inline__ __host__ cudaError_t cudaMalloc(
  T      **devPtr,
  size_t   size
)
*/

  cudaError_t err;
  err = cudaMalloc((void**)&d_A, mem_size);
  if (err != cudaSuccess) {
    err_quit("failed to allocate device vector A, error: %s", cudaGetErrorString(err));
  }
  err = cudaMalloc((void**)&d_B, mem_size);
  if (err != cudaSuccess) {
    err_quit("failed to allocate device vector B, error: %s", cudaGetErrorString(err));
  }
  err = cudaMalloc((void**)&d_C, mem_size);
  if (err != cudaSuccess) {
    err_quit("failed to allocate device vector C, error: %s", cudaGetErrorString(err));
  }

  /*
  extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
  */

  // 2. h2d
  err = cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    err_quit("failed to copy vector A from host memory to device memory, error: %s", cudaGetErrorString(err));
  }
  err = cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    err_quit("failed to copy vector B from host memory to device memory, error: %s", cudaGetErrorString(err));
  }

  // 3. perform computation on GPU
  size_t warp_size = warpSize;
  size_t threads_per_block = 256;
  size_t blocks_per_grid = (num_elements + threads_per_block - 1)/threads_per_block;
  printf("kernel lauch parameters: blocks_per_grid: %zu, threads_per_block: %zu, warpSize: %zu\n", blocks_per_grid, threads_per_block, warp_size);
  vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, num_elements);


  // 4. d2h: copy computation result from device memory to host memory
  err = cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    err_quit("failed to copy vector C from device memory to host memory, error: %s", cudaGetErrorString(err));
  }

  // 4.1 verify that the result vector is correct
  for (int i=0; i<num_elements; i++) {
    if (fabs(h_A[i]+h_B[i]-h_C[i]) > 1e-5) {
      err_quit("Result verification failed at element %d", i);
    }
  }
  printf("Test PASSED\n");

  // 5. destroy device and host memory
  /*
    cudaError_t cudaFree(void *devPtr)
  */
  err = cudaFree(d_A);
  if (err != cudaSuccess) {
    err_quit("failed to destroy device memory for vector A, error: %s", cudaGetErrorString(err));
  }
  err = cudaFree(d_B);
  if (err != cudaSuccess) {
    err_quit("failed to destroy device memory for vector B, error: %s", cudaGetErrorString(err));
  }
  err = cudaFree(d_C);
  if (err != cudaSuccess) {
    err_quit("failed to destroy device memory for vector C, error: %s", cudaGetErrorString(err));
  }
  free(h_A);
  free(h_B);
  free(h_C);
}