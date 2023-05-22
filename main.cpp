// a program that adds 1 to all elements of an array
// each GPU/Accelerator/CPU thread computes 4 element
// both input and output arrays are integer arrays
// 256k threads compute 1M elements

#include <iostream>

// uncomment this if you use opencl v2.0 or v3.0 devices. By default, opencl v1.2 devices are queried. 
// must be defined before including "gpgpu.hpp"
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include "gpgpu.hpp"
int main()
{
    try
    {
        constexpr size_t n = 1024*1024;
        int clonesPerDevice = 1;
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL,GPGPU::Computer::DEVICE_SELECTION_ALL,clonesPerDevice);
        GPGPU_LIB::PlatformManager man;

        auto deviceNames = computer.deviceNames();
        for (auto d : deviceNames)
        {
            std::cout << d << std::endl;
        }

        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Starting compilation of kernel..." << std::endl;

        computer.compile(std::string(R"(
                #define n )") + std::to_string(n) + std::string(R"(
                    
                    void kernel fmaTest(global float * a, global float * b) 
                    {
                        // global id of this thread
                        const int id = get_global_id(0);
                        const int localId = id % 256;
                        float r1=0.0f;
                        float r2=0.0f;
                        float r3=0.0f;
                        float r4=0.0f;
                        float a1=a[id];
                        float a2=a[(id+1)%n];
                        float a3=a[(id+2)%n];
                        float a4=a[(id+3)%n];
                        local float tmp[256];
                        for(int i=0;i<n;i+=256)
                        {
                            tmp[localId] = a[i];
                            barrier(CLK_LOCAL_MEM_FENCE);
                            for(int j=0;j<256;j++)
                            {
                                float r0 = tmp[j];
                                r1 = fma(a1,r0,r1);
                                r2 = fma(a2,r0,r2);
                                r3 = fma(a3,r0,r3);
                                r4 = fma(a4,r0,r4);
                            }
                            
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        b[id] = r1+r2+r3+r4;
                        
                    }
           )"), "fmaTest");


        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Starting allocation of host buffer..." << std::endl;

        // create parameters of kernel (also allocated in each device)
        bool isAinput = true;
        bool isBinput = false;
        bool isAoutput = false;
        bool isBoutput = true;
        bool isInputRandomAccess = true;
        int dataElementsPerThread = 1;
        GPGPU::HostParameter a = computer.createHostParameter<float>("a", n, dataElementsPerThread, isAinput, isAoutput, isInputRandomAccess);
        GPGPU::HostParameter b = computer.createHostParameter<float>("b", n, dataElementsPerThread, isBinput, isBoutput, false);

        // init elements of parameters
        for (int i = 0; i < n; i++)
        {
            a.access<float>(i) = i;
            b.access<float>(i) = 0;
        }
       
        // set kernel parameters (0: first parameter of kernel, 1: second parameter of kernel)
        computer.setKernelParameter("fmaTest", "a", 0);
        computer.setKernelParameter("fmaTest", "b", 1);

        int repeat = 100;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Starting computation for "<< repeat <<" times..." << std::endl;

        // copies input elements (a) to devices, runs kernel on devices, copies output elements to RAM (b), uses n/4 total threads distributed to devices, 256 threads per work-group in devices
        // faster devices are given more threads automatically (after every call to run method)
        size_t nano;
        
        std::vector<double> workloadRatios;
        {
            GPGPU::Bench bench(&nano);
            for (int i = 0; i < repeat; i++)
            {
                workloadRatios = computer.run("fmaTest", 0, n , 256); // n/4 number of total threads, 256 local threads per work group
            }
        }
        std::cout << nano / 1000000000.0 << " seconds" << std::endl;
        std::cout << (((repeat *(double) n * (double)n * 8 ) / (nano / 1000000000.0))/1000000000.0) << " gflops" << std::endl;
        for (int i = 0; i < deviceNames.size(); i++)
        {
            std::cout << deviceNames[i] << " has workload ratio of: " << workloadRatios[i] << std::endl;
        }

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}
