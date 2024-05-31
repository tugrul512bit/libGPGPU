// add 1 to all elements of a vector of length 64
// single-gpu version

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"


int main()
{
    try
    {

        const size_t n = 64;

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS,0/*select only first gpu*/);
        computer.compile(
            R"(

            kernel void vecAdd(global int * A, const global int * B, const int scalar) 
            { 
                const int threadId=get_global_id(0); 
                A[threadId] = B[threadId] + scalar;
            }

            )", "vecAdd");

        auto data1 =  computer.createArrayOutputAll<int>("A", n); // writes all results and assumes single gpu is used
        auto data2 =  computer.createArrayInput<int>("B", n); // loads all elements (broadcasts to all selected gpus)
        auto scalar = computer.createScalarInput<int>("scalar"); // this is not an array on GPU. its a kernel scalar value.
        auto kernelParams = data1.next(data2).next(scalar);
        
        for (int i = 0; i < 5; i++)
        {
            data2.access<int>(15)=i;
            scalar = 1000; // same as scalar.access<int>(0)=1000;
            computer.compute(kernelParams, "vecAdd", 0, 64 /* kernel threads */, 4 /* block threads */);                                
            std::cout << data1.access<int>(15)<<std::endl;
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; 
    }
    return 0;
}
