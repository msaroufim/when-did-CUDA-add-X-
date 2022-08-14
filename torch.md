# torch.cuda

https://pytorch.org/docs/stable/cuda.html

Supported features
* Default float32 operations to tfloat32
* Default to fp16 reductions
* Operations are asynchronous so to time operations leverage `torch.cuda.Event()`
* CUDA streams: A sequence of operations that belong to a specific device
* PyTorch uses a caching allocator to manage memory on GPU that may mess up results on `nvidia-smi`
* JIT kernels are stored in local cache, first time compilation takes time
* Pinned memory buffers help make data loading faster
* DistributeDataParallel uses multiple processes instead of multiplee threads which helps deal with contention of resources on GPU
* CUDA graph supported with partial and whole graph capture to bundle multiple small kernels into one large one to help with python and CPU overhead