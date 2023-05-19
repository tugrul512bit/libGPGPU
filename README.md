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

        // GPGPU::Computer::DEVICE_GPUS or _CPUS or _ACCS or "|" combination of them
        // GPGPU::Computer::DEVICE_SELECTION_ALL gets all selected devices, 0 to N selects only 1 of them (0-based indexing)
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL, GPGPU::Computer::DEVICE_SELECTION_ALL);

        // first compile a kernel (for now, it is in C99 language specs and compiled for every selected device)
        computer.compile(std::string(R"(

        #define n )") + std::to_string(n) + std::string(R"(

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

        // allocate input & output arrays, each with n integers, each with 4 elements per thread storage during load-balancing

        GPGPU::HostParameter a = computer.createHostParameter<int>("a", n, 4, true /* read input elements from RAM to VRAM*/, false, false /* true: read all elements, false: read only own lanes */);
        GPGPU::HostParameter b = computer.createHostParameter<int>("b", n, 4, false, true /* write output elements from VRAM to RAM */, false);

        for (int i = 0; i < n; i++)
        {
            a.access<int>(i) = i;
            b.access<int>(i) = 0;
        }


        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "a", 0);
        computer.setKernelParameter("add1ToEveryElementBut4ElementsPerThread", "b", 1);


        constexpr int repeat = 100;
        size_t nano, nano2 = 0;
        {

            for (int i = 0; i < repeat; i++)
            {
                {
                    Bench bench(&nano);
                    
                    // multi-step load-balancing: slowly converges to balance, low latency per step
                    computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256);
                    // this version does not need "setKernelParameter":
                    //computer.compute(a.next(b), "add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256);
                    
                    // in-step load balancing: good starting balance, fair balancing for uneven workloads, high latency
                    //computer.runFineGrainedLoadBalancing("add1ToEveryElementBut4ElementsPerThread", 0, n/4, 256,1024*8);
                    // this version does not need "setKernelParameter":
                    //computer.compute(a.next(b), "add1ToEveryElementBut4ElementsPerThread", 0, n/4, 256,true,1024*8);
                    
                }
                nano2 += nano;
                std::cout << "1 iteration = " << nano / 1000000000.0 << " seconds" << std::endl;

            }

        }

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
