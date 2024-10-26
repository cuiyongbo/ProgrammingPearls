/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wA         width of matrix A
//! @param hB         height of matrix B
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void ComputeGold(float *C, const float *A, const float *B, uint32_t hA, uint32_t wA, uint32_t hB, uint32_t wB) {
  assert(wA == hB);
  for (uint32_t i=0; i<hA; ++i) {
    for (uint32_t j=0; j<wB; ++j) {
      double sum = 0;
      for (int k=0; k<wA; k++) {
        //sum += A[i][k] * B[k][j]
        sum += A[i*wA+k] * B[k*wB+j];
      }
      //C[i][j] = sum;
      C[i*wB+j] = sum;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set on GPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wA         width of matrix A
//! @param hB         height of matrix B
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
__global__ void MatMulKernel_naive(float *C, const float *A, const float *B, uint32_t hA, uint32_t wA, uint32_t hB, uint32_t wB) {
  assert(wA == hB);
  float sum = 0;
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t k=0; k<wA; ++k) {
    //sum += A[i][k] * B[k][j];
    sum += A[row * wA + k] * B[k * wB + column];
  }
  //C[i][j] = sum;
  C[row * wB + column] = sum;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
    float *B, int wA,
    int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;
  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA;
  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;
  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;
  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread, it is private to each thread, and may store on registers
  // and we still iterate for `wA` time to calculate it: (aEnd-aBegin) / aStep * BLOCK_SIZE
  float Csub = 0;
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a < aEnd;
       a += aStep, b += bStep) {
    // shared by threads in a thread block
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  C[row * wB + column] = Csub;
}


void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
/*
CUDA programm ABC:
    1. prepare host and device memory
    2. h2d: copy vars from host memory to device memory
    3. perform computation on GPU
    4. d2h: copy computation result from device memory to host memory
    5. destroy device and host memory
*/

  // 1. prepare host and device memory
  printf("1. prepare host and device memory\n");
  uint32_t size_A = dimsA.x * dimsA.y;
  uint32_t mem_size_A = size_A * sizeof(float);
  uint32_t size_B = dimsB.x * dimsB.y;
  uint32_t mem_size_B = size_B * sizeof(float);
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  uint32_t size_C = dimsC.x * dimsC.y;
  uint32_t mem_size_C = size_C * sizeof(float);

  /*
  template<class T>
  static __inline__ __host__ cudaError_t cudaMallocHost(
    T            **ptr,
    size_t         size,
    unsigned int   flags = 0
  )
  */
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaMalloc((void**)&d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void**)&d_B, mem_size_B));
  checkCudaErrors(cudaMalloc((void**)&d_C, mem_size_C));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // 2. h2d: copy vars from host memory to device memory
  printf("2. h2d: copy vars from host memory to device memory\n");
  /*
  cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = (cudaStream_t)0)
  */
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
  
  // 3. perform computation on GPU
  printf("3. perform computation on GPU\n");
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x/threads.x, dimsA.y/threads.y);

  // warmup
  if (block_size == 16) {
    MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else {
    MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  printf("warmup done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // create CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  // record the start event
  checkCudaErrors(cudaEventRecord(start, stream));
  // profile kernel performance
  int n_iter = 300;
  for (int i=0; i<n_iter; ++i) {
    if (block_size == 16) {
      MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
      MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }
  // record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));
  // wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  // calculate average latency
  float ms_total = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&ms_total, start, stop));
  float ms_per_mm = ms_total/n_iter;
  // to compute C[i][j] we have to perform multiplication wA times and addition wA-1 times (about 2*wA times ops)
  // so the total ops would be (2*wA * hA*wB) times
  double flops_per_mm = 2.0 * dimsA.x * dimsA.y * dimsB.x;
  double giga_flops = (flops_per_mm * 1e-9f) / (ms_per_mm/1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      giga_flops, ms_per_mm, flops_per_mm, threads.x * threads.y);

  // 4. d2h: copy computation result from device memory to host memory
  printf("4. d2h: copy computation result from device memory to host memory\n");
  checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  // 4.1 verify result
  bool correct = true;

  /*
  // 4.1.1 verify result with CPU computation result
  float *ref_C;
  checkCudaErrors(cudaMallocHost((void**)&ref_C, mem_size_C));
  ComputeGold(ref_C, h_A, h_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
  double eps = 1e-5;
  for (uint32_t i=0; i<size_C; ++i) {
    if (fabs(h_C[i] - ref_C[i]) > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
              i, h_C[i], ref_C[i], eps);
      correct = false;
    }
  }
  checkCudaErrors(cudaFreeHost(ref_C));
  */

  // 4.1.2 verify result with GPU computation result
  float *ref_C, *ref_d_C;
  checkCudaErrors(cudaMallocHost((void**)&ref_C, mem_size_C));
  checkCudaErrors(cudaMalloc((void**)&ref_d_C, mem_size_C));
  MatMulKernel_naive<<<grid, threads, 0, stream>>>(ref_d_C, h_A, h_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
  checkCudaErrors(cudaMemcpyAsync(ref_C, ref_d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));
  double eps = 1e-6;
  for (uint32_t i=0; i<size_C; ++i) {
    if (fabs(h_C[i] - ref_C[i]) > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
              i, h_C[i], ref_C[i], eps);
      correct = false;
    }
  }
  checkCudaErrors(cudaFree(ref_d_C));
  checkCudaErrors(cudaFreeHost(ref_C));

  /*
  // test relative error by the formula: |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1e-6;
  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); ++i) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }
  */

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // 5. destroy device and host memory
  printf("5. destroy device and host memory\n");
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}


/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}


/*

The provided CUDA kernel code performs matrix multiplication using a tiled approach, which is efficient for GPU execution. Let's break down how the kernel works and how it maps threads to compute elements of the resulting matrix \( C \).

### Kernel Launch Configuration

The kernel is launched with a grid of thread blocks, where each block is of size `BLOCK_SIZE x BLOCK_SIZE`. The grid dimensions are calculated based on the dimensions of the input matrices \( A \) and \( B \):

```cpp
dim3 threads(block_size, block_size);
dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
```

Here, `dimsB.x` is the width of matrix \( B \) and `dimsA.y` is the height of matrix \( A \). Each block of threads will compute a `BLOCK_SIZE x BLOCK_SIZE` sub-matrix of the resulting matrix \( C \).

### Matrix Multiplication Kernel

The kernel `MatrixMulCUDA` performs the matrix multiplication using shared memory to optimize memory access patterns. Here's a step-by-step explanation of the kernel:

#### 1. Thread and Block Indices

Each thread block is responsible for computing a sub-matrix of \( C \):

```cpp
int bx = blockIdx.x; // Block index in the x direction
int by = blockIdx.y; // Block index in the y direction
int tx = threadIdx.x; // Thread index within the block in the x direction
int ty = threadIdx.y; // Thread index within the block in the y direction
```

#### 2. Compute Starting Points

Calculate the starting points for the sub-matrices of \( A \) and \( B \) that this block will work on:

```cpp
int aBegin = wA * BLOCK_SIZE * by; // Starting index of the sub-matrix of A
int aEnd = aBegin + wA - 1; // Ending index of the sub-matrix of A
int aStep = BLOCK_SIZE; // Step size to move to the next sub-matrix of A

int bBegin = BLOCK_SIZE * bx; // Starting index of the sub-matrix of B
int bStep = BLOCK_SIZE * wB; // Step size to move to the next sub-matrix of B
```

#### 3. Initialize Accumulator

Initialize an accumulator for the element of \( C \) that this thread will compute:

```cpp
float Csub = 0;
```

#### 4. Loop Over Sub-matrices

Loop over the sub-matrices of \( A \) and \( B \) to compute the product:

```cpp
for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load sub-matrices into shared memory
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to ensure all threads have loaded their sub-matrices
    __syncthreads();

    // Perform the multiplication for the sub-matrix
    for (int k = 0; k < BLOCK_SIZE; k++) {
        Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to ensure all threads have completed the computation
    __syncthreads();
}
```

#### 5. Write Result to Global Memory

Finally, write the computed value to the appropriate position in the output matrix \( C \):

```cpp
int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB * ty + tx] = Csub;
```

### Summary

- **Thread Mapping**: Each thread computes one element of the resulting matrix \( C \).
- **Shared Memory**: Sub-matrices of \( A \) and \( B \) are loaded into shared memory to reduce global memory access latency.
- **Tiling**: The kernel processes the matrices in tiles of size `BLOCK_SIZE x BLOCK_SIZE`, iterating over the necessary sub-matrices to compute the final result.
- **Synchronization**: `__syncthreads()` is used to ensure all threads in a block have completed their memory loads and computations before proceeding to the next step.

This approach leverages the parallelism of the GPU and the fast shared memory to efficiently perform matrix multiplication.


Q: when iterating over submatrix `As`, `Bs`, I see each thread loads one element every time, but the inner loop to calculate `Csub` would iterate over `BLOCK_SIZE` elements,
so the calculation cannot start untill the summatrices are all ready. does it mean `As`, `Bs` is shared by threads in one thread block?

Yes, you are correct. The `As` and `Bs` arrays are declared as `__shared__` memory, which means they are shared among all threads within a single thread block. This shared memory is used to store sub-matrices of \( A \) and \( B \) that are being processed by the threads in the block.

### Detailed Explanation

#### Shared Memory

Shared memory in CUDA is a type of memory that is shared among all threads in a block. It is much faster than global memory and is used to optimize memory access patterns and reduce latency.

#### Loading Sub-matrices into Shared Memory

Each thread in the block loads one element of the sub-matrix from global memory into shared memory. This is done in the following lines:

```cpp
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

// Load the matrices from device memory to shared memory
As[ty][tx] = A[a + wA * ty + tx];
Bs[ty][tx] = B[b + wB * ty + tx];

// Make sure the sub-matrices are loaded
__syncthreads();
```

Here, `ty` and `tx` are the thread indices within the block, so each thread loads one element of the sub-matrix into the shared memory arrays `As` and `Bs`.

#### Synchronization

The `__syncthreads()` function is a barrier synchronization function that ensures all threads in the block have completed their memory loads before any thread proceeds to the computation. This is crucial because the computation relies on the entire sub-matrix being available in shared memory.

#### Computation Using Shared Memory

Once the sub-matrices are loaded into shared memory and all threads have synchronized, each thread can then proceed to compute its part of the result matrix \( C \). The inner loop iterates over the elements of the sub-matrices:

```cpp
for (int k = 0; k < BLOCK_SIZE; k++) {
    Csub += As[ty][k] * Bs[k][tx];
}
```

In this loop:
- `As[ty][k]` accesses the elements of the sub-matrix of \( A \) along the row corresponding to the thread's `ty` index.
- `Bs[k][tx]` accesses the elements of the sub-matrix of \( B \) along the column corresponding to the thread's `tx` index.

Each thread computes a partial sum for the element of \( C \) it is responsible for. The `Csub` variable accumulates this partial sum.

#### Writing the Result to Global Memory

After all iterations over the sub-matrices are complete, each thread writes its computed value to the appropriate position in the output matrix \( C \):

```cpp
int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB * ty + tx] = Csub;
```

### Summary

- **Shared Memory**: `As` and `Bs` are shared memory arrays that are accessible by all threads within a block.
- **Loading Sub-matrices**: Each thread loads one element of the sub-matrix into shared memory.
- **Synchronization**: `__syncthreads()` ensures all threads have loaded their elements before any computation begins.
- **Computation**: Each thread computes a partial sum for its corresponding element in the result matrix \( C \) using the sub-matrices in shared memory.
- **Writing Result**: The computed value is written back to global memory.

This approach leverages the fast access speed of shared memory and the parallel processing capabilities of the GPU to efficiently perform matrix multiplication.

Q: since `Csub` accumulates partial sum which takes several loops to finish, is it a local variable for each thread? where does it reside, in register or shared memory, or global memory?

Yes, you are correct. The `As` and `Bs` arrays are declared as `__shared__` memory, which means they are shared among all threads within a single thread block. This shared memory is used to store sub-matrices of \( A \) and \( B \) that are being processed by the threads in the block.

### Detailed Explanation

#### Shared Memory

Shared memory in CUDA is a type of memory that is shared among all threads in a block. It is much faster than global memory and is used to optimize memory access patterns and reduce latency.

#### Loading Sub-matrices into Shared Memory

Each thread in the block loads one element of the sub-matrix from global memory into shared memory. This is done in the following lines:

```cpp
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

// Load the matrices from device memory to shared memory
As[ty][tx] = A[a + wA * ty + tx];
Bs[ty][tx] = B[b + wB * ty + tx];

// Make sure the sub-matrices are loaded
__syncthreads();
```

Here, `ty` and `tx` are the thread indices within the block, so each thread loads one element of the sub-matrix into the shared memory arrays `As` and `Bs`.

#### Synchronization

The `__syncthreads()` function is a barrier synchronization function that ensures all threads in the block have completed their memory loads before any thread proceeds to the computation. This is crucial because the computation relies on the entire sub-matrix being available in shared memory.

#### Computation Using Shared Memory

Once the sub-matrices are loaded into shared memory and all threads have synchronized, each thread can then proceed to compute its part of the result matrix \( C \). The inner loop iterates over the elements of the sub-matrices:

```cpp
for (int k = 0; k < BLOCK_SIZE; k++) {
    Csub += As[ty][k] * Bs[k][tx];
}
```

In this loop:
- `As[ty][k]` accesses the elements of the sub-matrix of \( A \) along the row corresponding to the thread's `ty` index.
- `Bs[k][tx]` accesses the elements of the sub-matrix of \( B \) along the column corresponding to the thread's `tx` index.

Each thread computes a partial sum for the element of \( C \) it is responsible for. The `Csub` variable accumulates this partial sum.

#### Writing the Result to Global Memory

After all iterations over the sub-matrices are complete, each thread writes its computed value to the appropriate position in the output matrix \( C \):

```cpp
int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB * ty + tx] = Csub;
```

### Summary

- **Shared Memory**: `As` and `Bs` are shared memory arrays that are accessible by all threads within a block.
- **Loading Sub-matrices**: Each thread loads one element of the sub-matrix into shared memory.
- **Synchronization**: `__syncthreads()` ensures all threads have loaded their elements before any computation begins.
- **Computation**: Each thread computes a partial sum for its corresponding element in the result matrix \( C \) using the sub-matrices in shared memory.
- **Writing Result**: The computed value is written back to global memory.

This approach leverages the fast access speed of shared memory and the parallel processing capabilities of the GPU to efficiently perform matrix multiplication.
*/
