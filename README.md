# libGPGPU

Some computers have multiple OpenCL-capable devices such as an integrated-GPU, a discrete-GPU and a CPU. But they do not automatically join their compute power to do some work quickly. There is a need of algorithm to separate a work into pieces and send them to all those devices. This library is written for unification of all(or only selected) devices within system, to run OpenCL kernels with load-balancing to minimize running-times of kernels as if they are part of a single GPU. [See wiki for details](https://github.com/tugrul512bit/libGPGPU/wiki).

- When CPU is included as a device, it is partitioned to dedicate some of threads for other devices' I/O management (copying buffers, synchronizing their threads, etc).
- Each device is given a dedicated CPU thread that does independent scheduling/synchronization for high performance load-balancing.
- RAM-sharing devices are given mapping ability instead of copying during computations. Integrated GPUs and CPUs get full RAM bandwidth when running kernels.
- - Only CPU or only iGPU can use this feature at the same time because OpenCL spec does undefined behavior if multiple devices use same host pointer during mapping/unmapping
- - Preferably (and by default) CPU is given the feature by constructor because non-gaming APUs have more core power than shader power. Gamers should have ```giveDirectRamAccessToCPU=false```on constructor
- - CPU RAM-sharing devices also benefit good from CPU L3 cache (especially if it is bigger than dataset)
- Devices can be cloned for overlapping I/O/compute operations to decrease overall latency or increase throughput during load-balancing. CPU & iGPU are not cloned.

Dependency:

- Visual Studio with vcpkg (that auto-installs OpenCL for the project) ![vcpkg](https://github.com/tugrul512bit/libGPGPU/assets/23708129/4a064dcb-b967-478d-a15f-fc69f4e3e9ee)
- - Maybe works in Ubuntu without vcpkg too, just need explicitly linking of OpenCL libraries and headers
- OpenCL 1.2 runtime (s) [Intel's runtime can find CPUs of AMD processors too & run AVX512 on Ryzen 7000 series CPU cores] (multiple platforms are scanned for all devices)
- OpenCL device(s) like GTX 1050 ti graphics card, a new CPU that has teraflops of performance, integrated GPU, all at the same time can be used as a big unified GPU.
- C++17

Hello-world sample that adds two vectors of length 1024:

```C++
#include <iostream>
#include <fstream>

// uncomment this if you use opencl v2.0 or v3.0 devices. By default, opencl v1.2 devices are queried. 
// must be defined before including "gpgpu.hpp"
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include "gpgpu.hpp"
int main()
{
    try
    {
        const int n = 1024; // number of array elements to test

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL); // allocate all devices for computations
        for (auto& name : computer.deviceNames())
            std::cout << name << std::endl;

        // compile a kernel to do C=A*m+B for all elements
        computer.compile(R"(
            kernel void blendFunc(float multiplier, global float * A, global float * B, global float * C) 
            { 
                int id=get_global_id(0); 
                C[id] = A[id] * multiplier + B[id];
             })", "blendFunc");

        // create host arrays that will be auto-copied-to/from GPUs/CPUs/Accelerators before/after kernel runs
        auto multiplier = computer.createScalarInput<float>("multiplier");
        multiplier.access<float>(0) = 3.1415f;

        // array inputs can be load-balanced (k items per thread) type or broadcasted type (all elements accessed per thread)
        auto A = computer.createArrayInputLoadBalanced<float>("A", n);
        auto B = computer.createArrayInputLoadBalanced<float>("B", n);
        auto C = computer.createArrayOutput<float>("C", n);

        // initialize one element for testing
        A.access<float>(400) = 2.0f;
        B.access<float>(400) = -3.1415f;
        C.access<float>(400) = 0.0f; 

        // compute, uses all GPUs and other devices with load-balancing to give faster devices more job to minimize overall latency of kernel (including copy latency too)
        computer.compute(multiplier.next(A).next(B).next(C),"blendFunc",0,n,64);
        std::cout << "PI = " << C.access<float>(400) << std::endl;

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; // any error is handled here
    }
    return 0;
}


```

output:

```
Device 0: GeForce GT 1030 (OpenCL 1.2 CUDA ) [direct-RAM-access disabled]
Device 1: gfx1036 (OpenCL 2.0 AMD-APP (3444.0) )[has direct access to RAM] [direct-RAM-access disabled]
Device 2: AMD Ryzen 9 7900 12-Core Processor (OpenCL 3.0 (Build 0) )[has direct access to RAM]
PI = 3.1415
```

Kernel parameters can be selected in two different ways:

- Explicitly setting parameters for only once, then calling kernel for multiple times
```C++
computer.setKernelParameter("kernelName", "a", 0);
computer.setKernelParameter("kernelName", "b", 1);
computer.run("kernelName", 0, n , 64); 
computer.run("kernelName", 0, n , 64); 
computer.run("kernelName", 0, n , 64); 

```
- Method-chaining to build a parameter-list in one-line:
```C++
computer.compute(a.next(b),"kernelName", 0, n, 64); 
computer.compute(a.next(b),"kernelName", 0, n, 64); 
computer.compute(a.next(b),"kernelName", 0, n, 64); 
```

both versions are equivalent with a trivial amount of extra host latency on second version.

Load balancing has two versions:
- dynamic: a queue is filled with many small pieces of work, then all devices independently consume the queue until it is empty. this has good work-distribution quality but high latency due to multiple synchronizations
- static: work is divided into bigger chunks and they are directly sent to all devices. after each run, device performances are calculated and a new(and better) work-distribution ratio is found for next run.

Static load balancing: good for uniform work-loads over work-items / data elements (simple image-processing algorithms, nbody algorithm, string-searching, etc)
```C++
// sample system: iGPU with 128 shaders @ 2GHz, dGPU with 384 shaders @ 1.5 GHz, CPU with 192 pipelines @ 5.3 GHz
computer.run("kernel", 0, n, 256); // equal work for all (50 milliseconds)
computer.run("kernel", 0, n, 256); // iGPU=1x work-items, dGPU=1.2x work-items, CPU=1.4x work-items (45 milliseconds)
computer.run("kernel", 0, n, 256); // iGPU=1x work-items, dGPU=1.5x work-items, CPU=2.0x work-items (33 milliseconds)
computer.run("kernel", 0, n, 256); // iGPU=1x work-items, dGPU=2.2x work-items, CPU=3.4x work-items (20 milliseconds)
computer.run("kernel", 0, n, 256); // iGPU=1x work-items, dGPU=2.4x work-items, CPU=3.7x work-items (17 milliseconds)
computer.run("kernel", 0, n, 256); // 15 milliseconds
computer.run("kernel", 0, n, 256); // 15 milliseconds
```

Dynamic load balancing: good for non-uniform work-loads (mandelbrot-set generation, ray tracing, etc)
```C++
// sample system: iGPU with 128 shaders @ 2GHz, dGPU with 384 shaders @ 1.5 GHz, CPU with 192 pipelines @ 5.3 GHz
// grain size = 2048 work-items (or 8x work-groups), can be any multiple of work group size
// local threads = 256 (work group size)
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds iGPU=1x work-items, dGPU=2.4x work-items, CPU=3.7x work-items (17 milliseconds)
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds
computer.runFineGrainedLoadBalancing("kernel", 0, n, 256,2048); // 20 milliseconds (with 5 milliseconds of extra sync-latency for queue-processing + 15 milliseconds of computation)
```
with this version, n work-items are divided into chunks of 2048 and are computed from a shared queue between all devices. Faster devices naturally take more chunks from queue and the work load is automatically balanced.
