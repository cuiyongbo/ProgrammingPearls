```
Q: is there only one thread grid to do computation when GPU lauches a kernel?

Yes, when a GPU launches a kernel, there is only one grid of thread blocks that is used to perform the computation. This grid is defined by the `blocks_per_grid` and `threads_per_block` parameters specified in the kernel launch configuration.

### Key Concepts

1. **Grid**: The grid is a collection of thread blocks. It is defined by the `blocks_per_grid` parameter.
2. **Block**: Each block is a collection of threads. It is defined by the `threads_per_block` parameter.
3. **Thread**: The smallest unit of execution within a block.

### Kernel Launch Configuration

When you launch a kernel, you specify the number of blocks in the grid and the number of threads in each block. For example:

```c++
dim3 threads_per_block(32, 32, 1);
dim3 blocks_per_grid((wB + threads_per_block.x - 1) / threads_per_block.x, (hA + threads_per_block.y - 1) / threads_per_block.y, 1);
matrix_multiplication<<<blocks_per_grid, threads_per_block>>>(d_C, d_A, d_B, wA, hA, wB, hB);
```

### Explanation

- **Threads per Block**: `dim3 threads_per_block(32, 32, 1)` specifies that each block contains \(32 \times 32 = 1024\) threads.
- **Blocks per Grid**: `dim3 blocks_per_grid((wB + threads_per_block.x - 1) / threads_per_block.x, (hA + threads_per_block.y - 1) / threads_per_block.y, 1)` specifies the number of blocks in the grid.

### Single Grid

When the kernel is launched, the GPU creates a single grid of thread blocks. This grid is used to perform the computation specified in the kernel function. All the threads within this grid execute the kernel code.

### Example

Consider the following example:

```c++
int wA = 200;
int hA = 200; 
int wB = 200;
int hB = 200;
assert(wA == hB);
float *d_C, *d_A, *d_B;
dim3 threads_per_block(32, 32, 1);
dim3 blocks_per_grid((wB + threads_per_block.x - 1) / threads_per_block.x, (hA + threads_per_block.y - 1) / threads_per_block.y, 1);
matrix_multiplication<<<blocks_per_grid, threads_per_block>>>(d_C, d_A, d_B, wA, hA, wB, hB);
```

### Calculation

- **Threads per Block**: \(32 \times 32 = 1024\) threads per block.
- **Blocks per Grid**: 
  - \( \text{blocks\_per\_grid}.x = \left\lceil \frac{wB}{\text{threads\_per\_block}.x} \right\rceil = \left\lceil \frac{200}{32} \right\rceil = 7 \)
  - \( \text{blocks\_per\_grid}.y = \left\lceil \frac{hA}{\text{threads\_per\_block}.y} \right\rceil = \left\lceil \frac{200}{32} \right\rceil = 7 \)
  - \( \text{blocks\_per\_grid}.z = 1 \)

- **Total Number of Blocks**: \(7 \times 7 \times 1 = 49\) blocks.
- **Total Number of Threads**: \(49 \times 1024 = 50176\) threads.

### Summary

- **Single Grid**: There is only one grid of thread blocks used to perform the computation when a kernel is launched.
- **Grid Configuration**: The grid configuration is defined by the `blocks_per_grid` and `threads_per_block` parameters.
- **Execution**: All threads within the grid execute the kernel code.

This single grid of thread blocks allows the GPU to efficiently parallelize the computation across many threads, leveraging the massive parallel processing capabilities of the GPU.

```