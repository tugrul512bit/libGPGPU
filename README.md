# libGPGPU

Multi-GPU, Multi-Accelerator and CPU device controller to run OpenCL kernels with load-balancing to minimize running-times of kernels. 

When CPU is included near other devices, the CPU device is partitioned to dedicate some of threads for other devices' I/O management (copying buffers, synchronizing their threads, etc).

Dependency:

- vcpkg (that auto-installs OpenCL to the project)

Hello-world sample:

```C++
// a program that adds 1 to all elements of an array
// each GPU/Accelerator/CPU thread computes 4 element
// both input and output arrays are integer arrays
// 256k threads compute 1M elements

#include <iostream>
#include "gpgpu.hpp"
int main()
{
    try
    {
        constexpr size_t n = 1024 * 1024;

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL, GPGPU::Computer::DEVICE_SELECTION_ALL);

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
        // uses only names of kernels and parameters to bind
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "a", 0);
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "b", 1);

        // copies input elements (a) to devices, runs kernel on devices, copies output elements to RAM (b), uses n/4 total threads distributed to devices, 256 threads per work-group in devices
        // faster devices are given more threads automatically
        computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // n/4 number of total threads, 256 local threads per work group

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
both versions are equivalent with a trivial amount of extra latency on second version.
