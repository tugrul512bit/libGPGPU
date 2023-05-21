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
        constexpr size_t n = 1024 * 1024;
        int clonesPerDevice = 1;
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL);
        GPGPU_LIB::PlatformManager man;
        man.printPlatforms();
        auto deviceNames = computer.deviceNames();
        for (auto d : deviceNames)
        {
            std::cout << d << std::endl;
        }

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
        size_t nano;
        {
            GPGPU::Bench bench(&nano);
            for (int i = 0; i < 1000; i++)
            {
                computer.run("add1ToEveryElementBut4ElementsPerThread", 0, n / 4, 256); // n/4 number of total threads, 256 local threads per work group
            }
        }
        std::cout << nano / 1000000000.0 << " seconds" << std::endl;

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
