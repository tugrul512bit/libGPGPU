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

        computer.compile(R"(
            kernel void vectorMul(global float * C, global float * D, global float * E) 
            { 
                int id=get_global_id(0); 
                E[id] = C[id] * D[id];
             })", "vectorMul");

        // create host arrays that will be auto-copied-to/from GPUs/CPUs/Accelerators before/after kernel runs
        // C = A + B
        // input, distributed to GPUs
        auto A = computer.createArrayInputLoadBalanced<float>("A",n);
        // input, distributed to GPUs
        auto B = computer.createArrayInputLoadBalanced<float>("B", n);
        // temporary state usage per-device
        auto C = computer.createArrayState<float>("C", n);

        // E = C*D
        // input, all elements broadcasted to GPUs
        auto D = computer.createArrayInput<float>("D", n);
        // output
        auto E = computer.createArrayOutput<float>("E", n);

        // initialize one element for testing
        A.access<float>(400) = 0.3f;
        B.access<float>(400) = 0.01415f;
        C.access<float>(400) = 0.0f; // this will be 0.31415 but not visible from host
        D.access<float>(400) = 10.0f;
        E.access<float>(400) = 0.0f; // this will be PI and as output will be visible

        

        // compute, uses all GPUs and other devices with load-balancing to give faster devices more job to minimize overall latency of kernel (including copy latency too)       
        for (int i = 0; i < 10; i++)
        {
            size_t nanoSeconds;
            {
                GPGPU::Bench bench(&nanoSeconds);
                // run 2 kernels with their own parameters (c=a+b for first, e=c*d for second kernel)
                // use n workitems total, 64 per work-group
                // runs both kernels on same command queue per device with same workload balance(and threads)
                computer.computeMultiple({ 
                        A.next(B).next(C),
                        C.next(D).next(E)
                    }, 
                    { 
                        "vectorAdd",
                        "vectorMul"
                    }, 0, n, 64);
            }
            std::cout << nanoSeconds << "ns" << std::endl;
        }
        std::cout << "PI = " << E.access<float>(400) << std::endl;

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; // any error is handled here
    }
    return 0;
}
