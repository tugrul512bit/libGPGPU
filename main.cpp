// video memory bandwidth test
// rtx4070: 
//      pcie read..: 22 GB/s 
//      pcie write.: 23 GB/s
//      pcie r+w...: 22 GB/s because program is not overlapping reads & writes on pcie for simplicity
//      vram read..: 470 GB/s
//      vram write.: 450 GB/s
//      vram r+w...: 430 GB/s
// requires 4GB video memory

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"
int main()
{
    try
    {
        bool testPCIEorVRAM = false; // false = VRAM
        bool testREAD = true;
        bool testWRITE = true;

        const size_t n = 1024ull*1024*512;

        // 0-index = first graphics card, no index(-1)=all gpus
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS,0);
        computer.compile(
            "#define N_BUFFER "+std::to_string(n)+"ull" +R"(
            )"+

            "#define TEST_PCIE " + std::to_string(testPCIEorVRAM) + R"(
            )" +
            "#define TEST_READ " + std::to_string(testREAD) + R"(
            )" +
            "#define TEST_WRITE " + std::to_string(testWRITE) + R"(
            )" +

            R"(

            kernel void bandwidth( 
                global int * data1,
                global int * data2) 
            { 
                const int threadId=get_global_id(0); 

                // in-kernel data copying to test only VRAM bandwidth
                if(!TEST_PCIE)
                {
                    int data=0;
                    if(TEST_READ && TEST_WRITE)
                        for(int i=0;i<N_BUFFER;i+=1024*1024)
                            data2[threadId + i]=data1[threadId + i];
                    else if(TEST_READ && !TEST_WRITE)
                    {
                        for(int i=0;i<N_BUFFER;i+=1024*1024)
                        {
                            data^=data1[threadId + i];
                        }
                        data2[threadId]=data;
                    }
                    else if(!TEST_READ && TEST_WRITE)
                        for(int i=0;i<N_BUFFER;i+=1024*1024)
                            data2[threadId + i]=0;

                    
                }



            }

        )", "bandwidth");

        // createArrayState does not make any pcie-transfer. its for keeping states within graphics card
        // createArrayInput: input data of kernel, copied from RAM to VRAM (all elements copied)
        // createArrayOutputAll: output data of kernel, copied from VRAM to RAM (all elements copied, not for multiple-GPUs due to race-condition on RAM buffer)
        auto data1 = (testPCIEorVRAM && testREAD) ? computer.createArrayInput<int>("data1", n) : computer.createArrayState<int>("data1", n);
        auto data2 = (testPCIEorVRAM && testWRITE) ? computer.createArrayOutputAll<int>("data2", n) : computer.createArrayState<int>("data2", n);
        auto kernelParams = data1.next(data2);

        // benchmark for 200 times
        for (int i = 0; i < 200; i++)
        {
            size_t nanoSeconds;
            {
                GPGPU::Bench bench(&nanoSeconds);                
                computer.compute(kernelParams, "bandwidth", 0, 1024*1024 /* kernel threads */, 1024 /* block threads */);                
            }
            if (testREAD)
                std::cout << "read ";
            if (testWRITE)
                std::cout << "write ";

            std::cout << (testREAD + testWRITE) * sizeof(int)* n / (double)nanoSeconds << " GB/s bandwidth"<<std::endl;
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; 
    }
    return 0;
}
