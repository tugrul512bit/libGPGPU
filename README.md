# libGPGPU

Multi-GPU, Multi-Accelerator and CPU device controller to run OpenCL kernels with load-balancing to minimize running-times of kernels. 

When CPU is included near other devices, the CPU device is partitioned to dedicate some of threads for other devices' I/O management (copying buffers, synchronizing their threads, etc).
