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

        // compile a kernel to do the adding C=A+B for all elmeents
        computer.compile(R"(
            kernel void vectorAdd(global float * A, global float * B, global float * C) 
            { 
                int id=get_global_id(0); 
                C[id] = A[id] + B[id];
             })", "vectorAdd");

        // create host arrays that will be auto-copied-to/from GPUs/CPUs/Accelerators before/after kernel runs
        auto A = computer.createArrayInputLoadBalanced<float>("A",n);
        auto B = computer.createArrayInputLoadBalanced<float>("B", n);
        auto C = computer.createArrayOutput<float>("C", n);

        // initialize one element for testing
        A.access<float>(400) = 3.0f;
        B.access<float>(400) = 0.1415f;
        C.access<float>(400) = 0.0f; // this will be PI

        // compute, uses all GPUs and other devices with load-balancing to give faster devices more job to minimize overall latency of kernel (including copy latency too)
       
        for (int i = 0; i < 10000; i++)
        {
            size_t nanoSeconds;
            {
                GPGPU::Bench bench(&nanoSeconds);
                computer.computeMultiple({ 
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C),
                        A.next(B).next(C)
                    }, 
                    { 
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd", 
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd",
                        "vectorAdd",
                    }, 0, n, 64);
            }
            std::cout << nanoSeconds << "ns" << std::endl;
        }
        std::cout << "PI = " << C.access<float>(400) << std::endl;

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; // any error is handled here
    }
    return 0;
}
