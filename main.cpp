

#include <vector>
#include <iostream>
#include <regex>
#include <fstream>
#include "gpgpu.hpp"
int main()
{
    try
    {
        constexpr size_t n = 1024*1024;
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_ALL, GPGPU::Computer::DEVICE_SELECTION_ALL);

        computer.compile(std::string(R"(

        #define n )") + std::to_string(n) + std::string(R"(

            void kernel fmaTest(global int * a, global int * b) 
            {

                // global id of this thread
                const int id = get_global_id(0);

       
                b[id] =  a[id*4] + a[id*4+1] + a[id*4+2] + a[id*4+3];
            }
           )"), "fmaTest");

        // true1: read mode (this is an input)
        // false: write mode 
        // true2: real whole array into device everytime a kernel with this parameter is run
        // 1: number of elements per work-item / gpu-thread
        // n: number of elements
        // "a": name of array 
        GPGPU::HostParameter a = computer.createHostParameter<int>("a", n, 4, true, false, false);
        GPGPU::HostParameter b = computer.createHostParameter<int>("b", n, 1, false, true, false);
        
        for (int i = 0; i < n; i++)
        {
            a.access<int>(i) = i;
            b.access<int>(i) = 0;
        }

        
        computer.setKernelParameter("fmaTest", "a", 0);
        computer.setKernelParameter("fmaTest", "b", 1);
        

        constexpr int repeat = 100;
        size_t nano, nano2 = 0;
        {

            for (int i = 0; i < repeat; i++)
            {
                {
                    Bench bench(&nano);
                    //computer.compute(a.next(b), "fmaTest", 0, n / 4, 256);
                    //computer.compute(a.next(b), "fmaTest", 0, n/4, 256,true,1024*8);
                    //computer.run("fmaTest", 0, n/4, 256);
                    computer.runFineGrainedLoadBalancing("fmaTest", 0, n/4, 256,1024*8);
                }
                nano2 += nano;
                std::cout << "1 iteration = " << nano/1000000000.0 << " seconds" << std::endl;

            }

        }


        std::cout << "nanoseconds = " << nano2 << std::endl;
        std::cout << "seconds = " << nano2 / 1000000000.0 << std::endl;
        std::cout << "giga add-operations per second = " << ( ((size_t) repeat)  * n / (double)(nano2 / 1000000000.0)) / 1000000000.0 << std::endl;
       
        // sum of all sub-sums
        size_t sum = 0;
        size_t realSum = 0;
        for (int i = 0; i < n; i++)
            realSum += i;
        for (int i = 0; i < n/4; i++)
        {
            sum += b.access<int>(i);
        }
        std::cout << "sum = " << sum << " real sum ="<<realSum << std::endl;
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}

