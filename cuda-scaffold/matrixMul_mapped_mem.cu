// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// a note: https://kharshit.github.io/blog/2024/06/07/matrix-multiplication-cuda

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
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < hA && column < wB) {
    float sum = 0;
    for (uint32_t k=0; k<wA; ++k) {
      //sum += A[i][k] * B[k][j];
      sum += A[row * wA + k] * B[k * wB + column];
    }
    //C[i][j] = sum;
    C[row * wB + column] = sum;
  }
}

template <int TILE_SIZE> __global__ void MatrixMulCUDAWithTile(float *C, float *A,
    float *B, int hA, int wA,
    int hB, int wB) {
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // element `C[row][column]` to be calculated
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;

  // shared memory arrays used to store sub-matrix
  // shared by threads in a thread block
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  // `Csub` is private to each thread, and may store on registers
  float Csub = 0;
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // and we still iterate for `wA` time to calculate `Csub`
  for (int m=0; m<(wA+TILE_SIZE-1)/TILE_SIZE; m++) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    if (row<hA && (m * TILE_SIZE + tx)<wA) {
      As[ty][tx] = A[row * wA + (m * TILE_SIZE + tx)];
    } else {
      As[ty][tx] = 0;
    }
    if ((m * TILE_SIZE + ty)<hB && column<wB) {
      Bs[ty][tx] = B[(m * TILE_SIZE + ty) * wB + column];
    } else {
      Bs[ty][tx] = 0;
    }
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    #pragma unroll
    for (int k=0; k<TILE_SIZE; k++) {
      Csub += As[ty][k] * Bs[k][tx];
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // write final result to device memory
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

  float *h_A, *h_B, *h_C;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_C, (void*)h_C, 0));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // 2. h2d: copy vars from host memory to device memory
  printf("2. h2d: copy vars from host memory to device memory\n");
  
  // 3. perform computation on GPU
  printf("3. perform computation on GPU\n");
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x/threads.x, dimsA.y/threads.y);

  // warmup
  if (block_size == 16) {
    MatrixMulCUDAWithTile<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
  } else {
    MatrixMulCUDAWithTile<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
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
      MatrixMulCUDAWithTile<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
    } else {
      MatrixMulCUDAWithTile<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
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
  // to compute C[i][j] we have to perform wA times multiplication and wA times addition
  // so the total ops would be (2*wA * hA * wB) times
  double flops_per_mm = 2.0 * dimsA.x * dimsA.y * dimsB.x;
  double giga_flops = (flops_per_mm * 1e-9f) / (ms_per_mm/1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      giga_flops, ms_per_mm, flops_per_mm, threads.x * threads.y);

  // 4. d2h: copy computation result from device memory to host memory
  printf("4. d2h: copy computation result from device memory to host memory\n");

  // 4.1 verify result
  bool correct = true;
  // 4.1.2 verify result with GPU computation result
  float *ref_C, *ref_d_C;
  checkCudaErrors(cudaMallocHost((void**)&ref_C, mem_size_C));
  checkCudaErrors(cudaHostGetDevicePointer((void**)&ref_d_C, (void*)ref_C, 0));
  MatMulKernel_naive<<<grid, threads, 0, stream>>>(ref_d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
  checkCudaErrors(cudaStreamSynchronize(stream));
  double eps = 1e-6;
  for (uint32_t i=0; i<size_C; ++i) {
    if (fabs(h_C[i] - ref_C[i]) > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
              i, h_C[i], ref_C[i], eps);
      correct = false;
    }
  }
  checkCudaErrors(cudaFreeHost(ref_C));

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // 5. destroy device and host memory
  printf("5. destroy device and host memory\n");
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
  int idev = findCudaDevice(argc, (const char **)argv);

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

  /*
  Set device to be used for GPU executions
  Sets device as the current device for the calling host thread. Valid device id's are 0 to (::cudaGetDeviceCount() - 1).
  */
  checkCudaErrors(cudaSetDevice(idev));
  /* To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling cudaSetDeviceFlags() with the cudaDeviceMapHost flag before any other CUDA call is performed. Otherwise, cudaHostGetDevicePointer() will return an error. */
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}

/* oops, ten times slower than case where we don't use mapped memory
root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# ./matrixMul 
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 208.38 GFlop/s, Time= 0.629 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.


root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# nsys profile --stat=true ./matrixMul_with_pinned_mem 
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 211.60 GFlop/s, Time= 0.619 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
Generating '/tmp/nsys-report-c947.qdstrm'
[1/8] [========================100%] report4.nsys-rep
[2/8] [========================100%] report4.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report4.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ---------------------
     74.9        538181456         15  35878763.7  13699794.0      2354  237448211   63369445.4  poll                 
     24.3        174815908        592    295297.1     32564.5       329   11347610     829704.2  ioctl                
      0.3          2293219         44     52118.6      6154.5      2195     845043     148239.2  open64               
      0.2          1539368         70     21991.0      1276.5       901    1376559     164266.2  fopen                
      0.1           782399         27     28977.7      4909.0      1997     451277      85381.8  mmap64               
      0.1           499688         21     23794.7      4456.0       665     202000      57568.0  mmap                 
      0.1           453643         10     45364.3     19221.0     17209     259822      75722.2  sem_timedwait        
      0.0           106182          2     53091.0     53091.0     51543      54639       2189.2  pthread_create       
      0.0            67799         12      5649.9      5685.0       570       9207       2644.2  write                
      0.0            61233         64       956.8       528.0       476      18011       2189.5  fclose               
      0.0            35275         25      1411.0        49.0        46      33964       6781.9  fgets                
      0.0            29805          6      4967.5      4949.5      1194       8933       2624.5  open                 
      0.0            20006          6      3334.3      3168.5      1887       5427       1271.5  munmap               
      0.0            17580         58       303.1       243.0       139        827        157.3  fcntl                
      0.0            12218          2      6109.0      6109.0      4502       7716       2272.6  socket               
      0.0            11607         15       773.8       435.0       323       2509        661.2  read                 
      0.0            10328          3      3442.7      3332.0      1181       5815       2319.0  pipe2                
      0.0            10262          1     10262.0     10262.0     10262      10262          0.0  connect              
      0.0             3977          1      3977.0      3977.0      3977       3977          0.0  bind                 
      0.0             3731          1      3731.0      3731.0      3731       3731          0.0  fread                
      0.0             3240         64        50.6        37.0        35        279         46.8  pthread_mutex_trylock
      0.0             1938          7       276.9       302.0       161        452        105.1  dup                  
      0.0              929          1       929.0       929.0       929        929          0.0  listen               
      0.0              535         10        53.5        46.0        45        118         22.7  fflush               

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)            Name           
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------------------------
     92.8        184803377          1  184803377.0  184803377.0  184803377  184803377          0.0  cudaEventSynchronize     
      3.5          6939966        302      22980.0       3047.0       2583    3525700     246396.9  cudaLaunchKernel         
      1.0          1978785          2     989392.5     989392.5     620291    1358494     521988.3  cudaStreamSynchronize    
      0.9          1748114          1    1748114.0    1748114.0    1748114    1748114          0.0  cudaProfilerStop         
      0.8          1533334          1    1533334.0    1533334.0    1533334    1533334          0.0  cudaMallocHost           
      0.7          1389194          4     347298.5     338687.5       3182     708637     394671.4  cudaFreeHost             
      0.4           788467          3     262822.3       3816.0       3749     780902     448670.2  cudaHostAlloc            
      0.0            24061          1      24061.0      24061.0      24061      24061          0.0  cudaStreamCreateWithFlags
      0.0            15052          1      15052.0      15052.0      15052      15052          0.0  cudaProfilerStart        
      0.0             6227          2       3113.5       3113.5       1767       4460       1904.2  cudaEventRecord          
      0.0             4989          2       2494.5       2494.5        547       4442       2754.2  cudaEventCreate          
      0.0             2379          1       2379.0       2379.0       2379       2379          0.0  cuModuleGetLoadingMode   
      0.0             2048          2       1024.0       1024.0        377       1671        915.0  cudaEventDestroy         

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     99.3        186252299        301   618778.4   618685.0    604732    635388       4682.9  void MatrixMulCUDAWithTile<(int)32>(float *, float *, float *, int, int, int, int)                  
      0.7          1357368          1  1357368.0  1357368.0   1357368   1357368          0.0  MatMulKernel_naive(float *, const float *, const float *, unsigned int, unsigned int, unsigned int,…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
SKIPPED: /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report4.sqlite does not contain GPU memory data.
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
SKIPPED: /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report4.sqlite does not contain GPU memory data.
Generated:
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report4.nsys-rep
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report4.sqlite
root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# 
*/