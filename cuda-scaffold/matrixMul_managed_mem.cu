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

  float *d_A, *d_B, *d_C;
  float *h_A, *h_B, *h_C;
  checkCudaErrors(cudaMallocManaged(&d_A, mem_size_A));
  checkCudaErrors(cudaMallocManaged(&d_B, mem_size_B));
  checkCudaErrors(cudaMallocManaged(&d_C, mem_size_C));
  h_A = d_A;
  h_B = d_B;
  h_C = d_C;
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

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
  checkCudaErrors(cudaMallocManaged((void**)&ref_d_C, mem_size_C));
  ref_C = ref_d_C;
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
  checkCudaErrors(cudaFree(ref_d_C));

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // 5. destroy device and host memory
  printf("5. destroy device and host memory\n");
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
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));
  if (!deviceProp.concurrentManagedAccess) {
    printf("Device %d does not fully support Managed Memory\n", idev);
    return 1;
  }

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}

/* basically no difference with case where we don't use managed memory
root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# ./matrixMul
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 1949.29 GFlop/s, Time= 0.067 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.


root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# nsys profile --stat=true ./matrixMul
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 1933.02 GFlop/s, Time= 0.068 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
Generating '/tmp/nsys-report-df28.qdstrm'
[1/8] [========================100%] report2.nsys-rep
[2/8] [========================100%] report2.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report2.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ---------------------
     70.0        548552070         22  24934185.0  10060914.5      3173  267166127   56251384.5  poll                 
     29.0        227431990        606    375300.3     97712.0       338   14540695    1083465.2  ioctl                
      0.5          4071155         44     92526.3      7721.5      2170    1780871     285386.5  open64               
      0.2          1533528         70     21907.5      1667.0       880    1361948     162502.8  fopen                
      0.1           802080         27     29706.7      6069.0      2520     465728      88119.1  mmap64               
      0.1           420015         10     42001.5     20432.5     14375     232145      66939.3  sem_timedwait        
      0.0           191770         21      9131.9      3477.0       697      79604      17443.1  mmap                 
      0.0           112755          2     56377.5     56377.5     46922      65833      13372.1  pthread_create       
      0.0            73370         12      6114.2      6370.0       567      11982       3084.4  write                
      0.0            65606         64      1025.1       558.0       473      19715       2401.5  fclose               
      0.0            34217         25      1368.7        55.0        47      32846       6557.8  fgets                
      0.0            30316          6      5052.7      4628.5      1126       9699       2920.6  open                 
      0.0            25442          9      2826.9      2811.0      1643       4254       1002.8  munmap               
      0.0            19076         58       328.9       274.5       135        816        164.3  fcntl                
      0.0            14310          2      7155.0      7155.0      5430       8880       2439.5  socket               
      0.0            13963         15       930.9       730.0       321       2514        687.6  read                 
      0.0            11410          3      3803.3      3610.0      1640       6160       2266.2  pipe2                
      0.0            10812          1     10812.0     10812.0     10812      10812          0.0  connect              
      0.0             4290          1      4290.0      4290.0      4290       4290          0.0  fread                
      0.0             3351          1      3351.0      3351.0      3351       3351          0.0  bind                 
      0.0             2962         64        46.3        36.0        34        247         36.5  pthread_mutex_trylock
      0.0             1829          7       261.3       284.0       165        385         89.0  dup                  
      0.0              933          1       933.0       933.0       933        933          0.0  listen               
      0.0              592         10        59.2        43.0        43        184         44.1  fflush               

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                Name               
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  ---------------------------------
     29.9         20569147          1  20569147.0  20569147.0  20569147  20569147          0.0  cudaProfilerStop                 
     29.9         20564640          4   5141160.0     26568.0      6638  20504866   10242482.3  cudaMallocManaged                
     28.1         19308233          1  19308233.0  19308233.0  19308233  19308233          0.0  cudaEventSynchronize             
      8.1          5601553        302     18548.2      3170.0      2768   3597993     214811.7  cudaLaunchKernel                 
      2.2          1545788          1   1545788.0   1545788.0   1545788   1545788          0.0  cudaGetDeviceProperties_v2_v12000
      1.4           984463          2    492231.5    492231.5    268205    716258     316821.3  cudaStreamSynchronize            
      0.2           117071          1    117071.0    117071.0    117071    117071          0.0  cudaFree                         
      0.1            36804          1     36804.0     36804.0     36804     36804          0.0  cudaStreamCreateWithFlags        
      0.0             6565          2      3282.5      3282.5      1924      4641       1921.2  cudaEventRecord                  
      0.0             3774          2      1887.0      1887.0       472      3302       2001.1  cudaEventCreate                  
      0.0             3731          1      3731.0      3731.0      3731      3731          0.0  cudaProfilerStart                
      0.0             2419          2      1209.5      1209.5       353      2066       1211.3  cudaEventDestroy                 
      0.0             2330          1      2330.0      2330.0      2330      2330          0.0  cuModuleGetLoadingMode           

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     98.7         20855804        301   69288.4   67072.0     66848    723451      37831.4  void MatrixMulCUDAWithTile<(int)32>(float *, float *, float *, int, int, int, int)                  
      1.3           266719          1  266719.0  266719.0    266719    266719          0.0  MatMulKernel_naive(float *, const float *, const float *, unsigned int, unsigned int, unsigned int,…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)               Operation              
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------------
     58.1           152485     34    4484.9    2831.0      2239     40384       6676.0  [CUDA memcpy Unified Host-to-Device]
     41.9           110128     24    4588.7    2367.5      1343     21119       5582.5  [CUDA memcpy Unified Device-to-Host]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)               Operation              
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------------
      2.097     24     0.087     0.031     0.004     0.520        0.147  [CUDA memcpy Unified Device-to-Host]
      2.097     34     0.062     0.016     0.004     1.032        0.180  [CUDA memcpy Unified Host-to-Device]

Generated:
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report2.nsys-rep
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report2.sqlite
*/