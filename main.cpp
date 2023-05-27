// hello-world program that blends A and B vectors

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

        // same as multiplier.access<float>(0) = 3.1415f;
        multiplier = 3.1415f;

        auto A = computer.createArrayInputLoadBalanced<float>("A", n);
        auto B = computer.createArrayInputLoadBalanced<float>("B", n);
        auto C = computer.createArrayOutput<float>("C", n);

        // initialize one element for testing
        A.access<float>(400) = 2.0f;
        B.access<float>(400) = -3.1415f;

        // initializing all elements at once
        C = 0.0f; 

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
