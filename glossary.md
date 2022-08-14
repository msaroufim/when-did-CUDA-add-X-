## CUDA glossary
https://docs.nvidia.com/cuda/cuda-c-programming-guide

## Typical working model
1. `malloc()` and `cudamalloc()` memory on CPU and GPU respectively
2. Copy data from CPU to GPU using `cudaMemcpy()` 
3. Asynchronously launch a kernel on GPU, threads can coordinate by using `__syncthreads()`
4. Copy result from GPU to hostt using `cudaMemcpy()`
5. Free memory for CPU and GPU using `free()` and `cudafree()` respectively

## Defining a kernel

`__global__` is how CUDA recongizes kernels and they can automatically leverage `threadIdx`, `blockIdx` and `blockDim`

```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```
## Main stuff

* Streaming Multiprocessor: The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs)
* CUDA grid: When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity
* Thread and Thread Block (or Warp): The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.
* PTX: CUDA's version of assembly
* CUDA context: The context holds all the management data to control and use the device. For instance, it holds the list of allocated memory, the loaded modules that contain device code, the mapping between CPU and GPU memory for zero copy, etc. https://stackoverflow.com/questions/43244645/what-is-a-cuda-context
* Compute Capability: determines the set of features supported by a given hardware it uses the convention `SM X.Y`. Each compute capability will vary the types of cores, their number, the number of schedulers, which custom functions are accelerated and will make some changes to how shared and global memory works


>Each SM contains 8 CUDA cores, and at any one time they're executing a single warp of 32 threads - so it takes 4 clock cycles to issue a single instruction for the whole warp. You can assume that threads in any given warp execute in lock-step, but to synchronise across warps, you need to use `__syncthreads()`. https://stackoverflow.com/questions/3519598/streaming-multiprocessors-blocks-and-threads-cuda

## Memory
* Local memory: Per thread private memory
* Shared memory: Per thread block memory, allows threads to communicate if you use `__syncthreads()`
* Global memory: Available to all threads
* Managed memory: Available to both GPU and host CPU

To speedup data transfers from CPU to GPU can leverage pinned memory to make sure memory on GPU doesn't get page locked, this cost a bunch of RAM to use

Can also give hints to leverage L2 cache

