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
        constexpr size_t n = 1024*16;
        int clonesPerDevice = 2;
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

                // algorithm from: https://github.com/sessamekesh/IndigoCS_Mandelbrot/blob/master/main.cpp

                #define n )") + std::to_string(n) + std::string(R"(
                    

                    int findMandelbrot(float cr, float ci, int max_iterations)
                    {
	                    int i = 0;
	                    float zr = 0.0f, zi = 0.0f;
	                    while (i < max_iterations && zr * zr + zi * zi < 4.0f)
	                    {
		                    float temp = zr * zr - zi * zi + cr;
		                    zi = 2.0f * zr * zi + ci;
		                    zr = temp;
		                    i++;
	                    }

	                    return i;
                    }

                    float mapToReal(int x, int imageWidth, float minR, float maxR)
                    {
	                    float range = maxR - minR;
	                    return x * (range / imageWidth) + minR;
                    }

                    float mapToImaginary(int y, int imageHeight, float minI, float maxI)
                    {
	                    float range = maxI - minI;
	                    return y * (range / imageHeight) + minI;
                    }

                    void kernel mandelbrot(global unsigned char * b) 
                    {
                        // global id of this thread
                        const int id = get_global_id(0);
                        const int x = id % n;
                        const int y = id / n;
			            float cr = mapToReal(x, n, -1.5f, 0.7f);
			            float ci = mapToImaginary(y, n, -1.0f, 1.0f);


			            int mn = findMandelbrot(cr, ci, 50);

                        b[id] = mn;
                        
                    }
           )"), "mandelbrot");


        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Starting allocation of host buffer..." << std::endl;

        // create parameters of kernel (also allocated in each device)

        bool isBinput = false;
        bool isBoutput = true;
        int dataElementsPerThread = 1;

        GPGPU::HostParameter b = computer.createHostParameter<unsigned char>("b", n*n, dataElementsPerThread, isBinput, isBoutput, false);

        // init elements of parameters
        for (int i = 0; i < n*n; i++)
        {
            b.access<unsigned char>(i) = 0;
        }


        int repeat = 100;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Starting computation for "<< repeat <<" times..." << std::endl;


        size_t nano;
        
        std::vector<double> workloadRatios;
        {
            GPGPU::Bench bench(&nano);
            for (int i = 0; i < repeat; i++)
            {
                std::cout << i << std::endl;
                workloadRatios = computer.compute(b,"mandelbrot", 0, n*n , 256,true,1024*1024); // n*n total workitems, 256 local workitems, 1024*1024 load-balancing grain size
            }
        }
        std::cout << nano / 1000000000.0 << " seconds" << std::endl;

        size_t totalIter = 0;
        for (int i = 0; i < n * n; i++)
        {
            totalIter += 8/* from mapping functions */ + b.access<unsigned char>(i) ;
        }
        totalIter *= 10;
        totalIter *= repeat;

        std::cout << (((totalIter ) / (nano / 1000000000.0))/1000000000.0) << " gflops" << std::endl;
        for (int i = 0; i < deviceNames.size(); i++)
        {
            std::cout << deviceNames[i] << " has workload ratio of: " << workloadRatios[i] << std::endl;
        }


        std::cout << "Creating 2GB ppm file. This may take a few minutes." << std::endl;
        std::ofstream fout("output_image.ppm");
        fout << "P3" << std::endl; 
        fout << n << " " << n << std::endl; 
        fout << "255" << std::endl; 

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fout << (int) b.access<unsigned char>(i*n+j) << " " << (int)b.access<unsigned char>(i * n + j) << " " << (int)b.access<unsigned char>(i * n + j) << " ";
            }
            fout << std::endl;
        }
        fout.close();

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}
