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
    MatrixMulCUDAWithTile<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
    //MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else {
    MatrixMulCUDAWithTile<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
    //MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
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
      //MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
      MatrixMulCUDAWithTile<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.y, dimsA.x, dimsB.y, dimsB.x);
      //MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
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
root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# ./matrixMul
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 1949.93 GFlop/s, Time= 0.067 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.


root@di-20241115115906-kfh5w:~/code/cuda-samples/Samples/0_Introduction/matrixMul# nsys profile --stat=true ./matrixMul_naive 
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ada" with compute capability 8.9

MatrixA(320,320), MatrixB(640,320)
1. prepare host and device memory
2. h2d: copy vars from host memory to device memory
3. perform computation on GPU
warmup done
Performance= 1933.64 GFlop/s, Time= 0.068 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
4. d2h: copy computation result from device memory to host memory
Result = PASS
5. destroy device and host memory

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
Generating '/tmp/nsys-report-8082.qdstrm'
[1/8] [========================100%] report3.nsys-rep
[2/8] [========================100%] report3.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report3.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  ---------------------
     67.3        371415742         13  28570441.7  3509819.0      5308  231068328   63406754.6  poll                 
     32.0        176249137        606    290840.2    29692.5       319   15406420    1055443.3  ioctl                
      0.3          1504652         70     21495.0     1202.5       897    1356030     161832.2  fopen                
      0.1           788079         27     29188.1     5853.0      1979     459632      87163.0  mmap64               
      0.1           495160         21     23579.0     4701.0       629     206482      57382.8  mmap                 
      0.1           409220         10     40922.0    19404.0     14006     231916      67208.7  sem_timedwait        
      0.1           366446         44      8328.3     5304.5      2135      80758      11995.7  open64               
      0.0           106931          2     53465.5    53465.5     50004      56927       4895.3  pthread_create       
      0.0            78650         12      6554.2     6712.5       483      15707       3751.8  write                
      0.0            59790         64       934.2      508.0       475      17753       2158.9  fclose               
      0.0            33411         25      1336.4       50.0        46      32143       6418.0  fgets                
      0.0            29570          6      4928.3     4503.5      1175       9290       2822.9  open                 
      0.0            18856          6      3142.7     3268.5      1697       4299        901.6  munmap               
      0.0            17763         58       306.3      228.5       137        940        172.1  fcntl                
      0.0            12988         15       865.9      564.0       332       2727        714.0  read                 
      0.0            12873          2      6436.5     6436.5      5412       7461       1448.9  socket               
      0.0            10955          3      3651.7     3272.0      1510       6173       2354.6  pipe2                
      0.0             9140          1      9140.0     9140.0      9140       9140          0.0  connect              
      0.0             4141          1      4141.0     4141.0      4141       4141          0.0  fread                
      0.0             3410         64        53.3       35.0        33        245         53.0  pthread_mutex_trylock
      0.0             2732          1      2732.0     2732.0      2732       2732          0.0  bind                 
      0.0             2052          7       293.1      336.0       177        438        101.0  dup                  
      0.0              918          1       918.0      918.0       918        918          0.0  listen               
      0.0              526         10        52.6       47.0        43        109         20.0  fflush               

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  -------------------------
     55.3         19286656          1  19286656.0  19286656.0  19286656  19286656          0.0  cudaEventSynchronize     
     17.2          5983438        302     19812.7      3161.5      2727   3488871     218066.3  cudaLaunchKernel         
      6.2          2162316          4    540579.0    481870.5      2597   1195978     627997.0  cudaFreeHost             
      5.4          1873992          1   1873992.0   1873992.0   1873992   1873992          0.0  cudaProfilerStop         
      5.4          1867303          4    466825.8    308310.0      2967   1247716     593638.4  cudaFree                 
      3.7          1298724          4    324681.0     81018.5      1444   1135243     545302.8  cudaMalloc               
      3.6          1264352          1   1264352.0   1264352.0   1264352   1264352          0.0  cudaMallocHost           
      2.3           812057          3    270685.7      4015.0      2105    805937     463542.2  cudaHostAlloc            
      0.7           230755          3     76918.3     72776.0     37578    120401      41566.6  cudaStreamSynchronize    
      0.1            41646          4     10411.5      9782.0      3045     19037       6633.8  cudaMemcpyAsync          
      0.1            24504          1     24504.0     24504.0     24504     24504          0.0  cudaStreamCreateWithFlags
      0.0             6218          2      3109.0      3109.0      1866      4352       1757.9  cudaEventRecord          
      0.0             5455          2      2727.5      2727.5       495      4960       3157.2  cudaEventCreate          
      0.0             2852          1      2852.0      2852.0      2852      2852          0.0  cudaProfilerStart        
      0.0             2465          1      2465.0      2465.0      2465      2465          0.0  cuModuleGetLoadingMode   
      0.0             2047          2      1023.5      1023.5       356      1691        944.0  cudaEventDestroy         

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     99.5         20197334        301   67100.8   67040.0     66720     76063        546.0  void MatrixMulCUDAWithTile<(int)32>(float *, float *, float *, int, int, int, int)                  
      0.5            94527          1   94527.0   94527.0     94527     94527          0.0  MatMulKernel_naive(float *, const float *, const float *, unsigned int, unsigned int, unsigned int,…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     55.2            65119      2   32559.5   32559.5     32447     32672        159.1  [CUDA memcpy Device-to-Host]
     44.8            52832      2   26416.0   26416.0     18176     34656      11653.1  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      1.638      2     0.819     0.819     0.819     0.819        0.000  [CUDA memcpy Device-to-Host]
      1.229      2     0.614     0.614     0.410     0.819        0.290  [CUDA memcpy Host-to-Device]

Generated:
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report3.nsys-rep
    /root/code/cuda-samples/Samples/0_Introduction/matrixMul/report3.sqlite
*/