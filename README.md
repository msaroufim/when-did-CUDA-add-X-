# When did CUDA add X?

## CUDA 11.0+

### CUB
https://nvlabs.github.io/cub/

CUB provides a couple of higher level software components that make it easier to use CUDA. They provide C++ templates and allow for compile time kernel tuning depending on the specific GPU that you're using.

### MIG
https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ 

MIG stands for multi instance GPU support it was introduced in Ampere as a way to partition a single GPU into 7 seven separate instances. As the on device memory of a GPU keeps increasing, this is a very practical feature for teams deploying many smaller each with low GPU utilization. The previous option had historically been virtualized GPU vGPU but given how they don't provide the same kind of workload isolation it's best to start using MIG when you can.

### nvJPEG2000
https://developer.nvidia.com/nvjpeg

This is an extension of the previous work on nvJPEG in CUDA 10.0 but the main difference being is that nvJPEG allows you to decode a specific subset of an image, a specific tile, has more image compression techniques. The main application NVIDIA pitches for this is large images like the ones regularly found in pathology. I couldn't find any benchmarks that explicitly compare nvJPEG to nvJPEG 2000 only one of those to CPU decoding libraries. 




 ## CUDA 10.0+

 ### nvJPEG
https://developer.nvidia.com/nvjpeg

At a high level it makes sense that GPUs handle this because they're already pretty good at decoding image data for video games and also given how memory bandwidth is the primary bottleneck for training models nowadays, it makes sense to speed the decoding process which is a major chunk of the general data loading process. 


## CUDA 9.0+

### CUTLASS 
https://github.com/NVIDIA/cutlass

CUTLASS provides a collection of C++ templates for high performance GEMM (GEneralized Matrix Multiplication). It provides support for mixed precision computation, leverages Tensor cores and can use thread-wide, warp-wide, block-wide and device-wide policies. This is an open source project so you can take a look at the many examples that NVIDIA provides.

## CUDA 8.0+

### nView
nView is a high performance desktop tile management software so you can use it to get GPU accelerated tile placements and quickly render different panels on your screen. This is kind of an odd library to include directly in CUDA since for other similar products like NVIDIA broadcast which allows for GPU accelerated background and noise removal for streaming those are included as separate libraries.

# NVWMI

NVWMI (NVIDIA enterprise Management toolkit) is a more relevant for IT admins that need to maintain a consistent set of configurations across a large fleet of GPUs so this won't be relevant to most but if you happen to be maintaining your own data center of GPUs and would like to for example have the same clock-speed across all of them, this is the tool for you.

## PhysX

Phyx is quite popular among game developers that need GPU accelerated physics ranging from soft body simulations (think cars crashing into each other), cloth, particle effects, fluids. Typically a given step of a game needs to run in less than 1/60s otherwise gamers will loudly complain. While PhysX is mostly used by game developers it can also be used Machine Learning researchers trying to develop faster simulations for RL engines.

## cuBLAS
https://developer.nvidia.com/cublas

cuBLAS provides GPU accelerated implementations of basic linear algebra subroutines (BLAS) so think things like scalar to vector to matrix multiplication.

## cuFFT
https://docs.nvidia.com/cuda/cufft/index.html

An accelerated implementation of Fast Fourier Transform which is useful in a couple of different domains ranging from audio/signal processing and image processing

## cuRAND
https://docs.nvidia.com/cuda/curand/index.html

Implementations for fast pseudorandom number of generation, there are many applications that benefit from randomness, a notable one for me is stochastic rounding which can help acccelerate the end to end training time of deep neural networks


## cuSPARSE
https://docs.nvidia.com/cuda/cusparse/index.html

GPU accelerated implementations of BLAS algorithms that work in sparse settings like a sparse to dense vector multiplication, sparse matrix to dense vector, a sparse matrix and a set of dense vectors. There are a couple of different known formats for sparse data like COO, CSR, CSC BSRX which are supported and useful in scenarios like sequence padding in language modeling or working with large graph data in deep learning.

## cuSOLVER
https://docs.nvidia.com/cuda/cusolver/index.html

A higher level library based on cuSPARSE and cuBLAS to solve a system of linear system of equations regardless of whether they're dense or sparse. This is useful in a wide variety of numerical and scientific applications

## NPP
https://developer.nvidia.com/npp

NVIDIA performance primitives provide support for fast GPU image, video and signal preprocessing like compression, filtering, thresholding, color conversion etc.. So if you're used to doing something like a bunch of transforms on your data before sending it to GPU you should consider doing it directly on GPU if memory bandwidth isn't your primary bottleneck.

## nvGRAPH
https://developer.nvidia.com/nvgraph

A graph analytics library that provides GPU acceleration for Page rank for recommenders, single source shortest path which is useful for a wide variety of algorithmic problems and single source widest path for scenarios like IP traffic routing.


## NVML
https://developer.nvidia.com/nvidia-management-library-nvml

NVIDIA Management library helps you measure the error counts, GPU utilization, clock speed, temperature and power management for your GPU. So pretty much anything you see when you type in nvidia-smi is instrumented via NVML.

## NVRTC
NVIDIA runtime compilation library takes in CUDA C++ source code and then helps you obtain PTX code which is the CUDA version of assembly.

