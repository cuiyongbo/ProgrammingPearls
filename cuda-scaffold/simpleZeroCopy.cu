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

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

int main(int argc, char **argv) {
  int deviceCount;
  int idev = 0;  // use default device 0
  char *device = NULL;
  unsigned int flags;
  float *a, *b, *c;           // Pinned memory allocated on the CPU
  float *d_a, *d_b, *d_c;     // Device pointers for mapped memory
  float errorNorm, refNorm, ref, diff;
  cudaDeviceProp deviceProp;

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Usage:  simpleZeroCopy [OPTION]\n\n");
    printf("Options:\n");
    printf("  --device=[device #]  Specify the device to be used\n");
    printf(
        "  --use_generic_memory (optional) use generic page-aligned for system "
        "memory\n");
    return EXIT_SUCCESS;
  }

  /* Get the device selected by the user or default to 0, and then set it. */
  if (getCmdLineArgumentString(argc, (const char **)argv, "device", &device)) {
    cudaGetDeviceCount(&deviceCount);
    idev = atoi(device);

    if (idev >= deviceCount || idev < 0) {
      fprintf(stderr,
              "Device number %d is invalid, will use default CUDA device 0.\n",
              idev);
      idev = 0;
    }
  }

  // if GPU found supports SM 1.2, then continue, otherwise we exit
  if (!checkCudaCapabilities(1, 2)) {
    exit(EXIT_SUCCESS);
  }

  printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));
  if (!deviceProp.canMapHostMemory) {
    fprintf(stderr, "Device %d does not support mapping CPU host memory!\n",
            idev);
    exit(EXIT_SUCCESS);
  }

  /*
  Set device to be used for GPU executions
  Sets device as the current device for the calling host thread. Valid device id's are 0 to (::cudaGetDeviceCount() - 1).
  */
  checkCudaErrors(cudaSetDevice(idev));

  /* To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling cudaSetDeviceFlags() with the cudaDeviceMapHost flag before any other CUDA call is performed. Otherwise, cudaHostGetDevicePointer() will return an error. */
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  /* Allocate mapped CPU memory. */
  int nelem = 1048576;
  size_t bytes = nelem * sizeof(float);

  /**
   * \brief \hl Allocates page-locked memory on the host
   *
   * Allocates \p size bytes of host memory that is page-locked and accessible
   * to the device. The driver tracks the virtual memory ranges allocated with
   * this function and automatically accelerates calls to functions such as
   * ::cudaMemcpy(). Since the memory can be accessed directly by the device, it
   * can be read or written with much higher bandwidth than pageable memory
   * obtained with functions such as ::malloc(). Allocating excessive amounts of
   * pinned memory may degrade system performance, since it reduces the amount
   * of memory available to the system for paging. As a result, this function is
   * best used sparingly to allocate staging areas for data exchange between host
   * and device.
   * 
   * The \p flags parameter enables different options to be specified that affect
   * the allocation, as follows.
   * - ::cudaHostAllocDefault: This flag's value is defined to be 0.
   * - ::cudaHostAllocPortable: The memory returned by this call will be
   * considered as pinned memory by all CUDA contexts, not just the one that
   * performed the allocation.
   * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
   * The device pointer to the memory may be obtained by calling
   * ::cudaHostGetDevicePointer().
  */
  flags = cudaHostAllocMapped;
  checkCudaErrors(cudaHostAlloc((void**)&a, bytes, flags));
  checkCudaErrors(cudaHostAlloc((void**)&b, bytes, flags));
  checkCudaErrors(cudaHostAlloc((void**)&c, bytes, flags));

  /* Initialize the vectors. */
  for (int n = 0; n < nelem; n++) {
    a[n] = rand() / (float)RAND_MAX;
    b[n] = rand() / (float)RAND_MAX;
  }

  /*
  Passes back the device pointer corresponding to the mapped, pinned host buffer allocated by ::cudaHostAlloc() or registered by ::cudaHostRegister().
  */
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_a, (void*)a, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_b, (void*)b, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void**)&d_c, (void*)c, 0));

  printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
  dim3 block(256);
  dim3 grid((uint32_t)ceil(nelem/(float)block.x));
  vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
  checkCudaErrors(cudaDeviceSynchronize());

  /* compare the result*/
  printf("> Checking the results from vectorAddGPU() ...\n");
  errorNorm = 0.0f;
  refNorm = 0.0f;
  for (int n=0; n<nelem; ++n) {
    ref = a[n] + b[n];
    diff = c[n] - ref;
    errorNorm += diff * diff;
    refNorm += ref * ref;
  }
  errorNorm = (float)sqrt((double)errorNorm);
  refNorm = (float)sqrt((double)refNorm);

  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFreeHost(b));
  checkCudaErrors(cudaFreeHost(c));

  exit(errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
