# libGPGPU

Multi-GPU, Multi-Accelerator and CPU device controller to run OpenCL kernels with load-balancing to minimize running-times of kernels. 

- When CPU is included as a device, it is partitioned to dedicate some of threads for other devices' I/O management (copying buffers, synchronizing their threads, etc).
- Each device is given a dedicated CPU thread that does independent scheduling/synchronization for high performance load-balancing.
- RAM-sharing devices are given mapping ability instead of copying during computations. Integrated GPUs and CPUs get full RAM bandwidth when running kernels.
- Devices can be cloned for overlapping I/O/compute operations to decrease overall latency or increase throughput during load-balancing. CPU is not cloned.

Dependency:

- vcpkg (that auto-installs OpenCL for the project)
- OpenCL 1.2 runtime (s) [Intel's runtime can find CPUs of AMD processors too & run AVX512 on Ryzen 7000 series CPU cores] (multiple platforms are scanned for all devices)
- OpenCL device(s) like GTX 1050 ti graphics card, Ryzen 7900x CPU, integrated GPU, all at the same time can be used as a big unified GPU.
- C++17

Hello-world sample:

```C++
// a program that adds 1 to all elements of an array, computed on all devices with a number of work-items given to them
// each GPU/Accelerator/CPU "work-item" computes 4 element
// both input and output arrays are integer arrays
// 256k work-items compute 1M elements (they flow through shader-pipelines in GPUs and SIMD units in CPUs)

#include <iostream>
#include "gpgpu.hpp"
int main()
{
    try
    {
        constexpr size_t n = 1024 * 1024;

        int clonesPerDevice = 1;
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL, GPGPU::Computer::DEVICE_SELECTION_ALL, clonesPerDevice);

        computer.compile(std::string(R"(
                #define n )") + std::to_string(n) + std::string(R"(
                    // C99 code for OpenCL 1.2 [C++ for OpenCL 2.0]
                    // every GPU thread adds one to 4 elements
                    void kernel add1ToEveryElementBut4ElementsPerThread(global int * a, global int * b) 
                    {
                        // global id of this thread
                        const int id = get_global_id(0);
                        const int dataStart = id*4;
                        for(int i=dataStart; i < dataStart + 4; i++)
                        {
                            b[i] = a[i] + 1;
                        }
                    }
           )"), "add1ToEveryElementBut4ElementsPerThread");

        // create parameters of kernel (also allocated in each device)
        bool isAinput = true;
        bool isBinput = false;
        bool isAoutput = false;
        bool isBoutput = true;
        bool isInputRandomAccess = false;
        int dataElementsPerThread = 4;
        GPGPU::HostParameter a = computer.createHostParameter<int>("a", n, dataElementsPerThread, isAinput, isAoutput, isInputRandomAccess);
        GPGPU::HostParameter b = computer.createHostParameter<int>("b", n, dataElementsPerThread, isBinput, isBoutput, isInputRandomAccess);

        // init elements of parameters
        for (int i = 0; i < n; i++)
        {
            a.access<int>(i) = i;
            b.access<int>(i) = 0;
        }

        // set kernel parameters (0: first parameter of kernel, 1: second parameter of kernel)
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "a", 0);
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "b", 1);

        // copies input elements (a) to devices, runs kernel on devices, copies output elements to RAM (b), uses n/4 total threads distributed to devices, 256 threads per work-group in devices
        // faster devices are given more threads automatically (after every call to run method)
        computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // n/4 number of total threads, 256 local threads per work group
        computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // balancing more
        computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // slowly converging to optimum balance where total computation time is minimized

        // check output array
        for (int i = 0; i < n; i++)
        {
            if (a.access<int>(i) + 1 != b.access<int>(i))
                throw std::invalid_argument("error: opencl did not work!");
        }
        std::cout << "ok" << std::endl;
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}

```

Kernel parameters can be selected in a different way by method-chaining:

```C++
// Replace this:
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "a", 0);
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "b", 1);
        computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // n/4 number of total threads, 256 local threads per work group
        
// With this:
        computer.compute(a.next(b),"add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); 
```
both versions are equivalent with a trivial amount of extra host latency (and less device-latency) on second version.

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
