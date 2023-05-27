#pragma once
#ifndef GPGPU_DEVICE_LIB
#define GPGPU_DEVICE_LIB


#include "gpgpu_init.hpp"
#include<iostream>
namespace GPGPU_LIB
{
	// wrapper for opencl device object with some queried device specs
	struct Device
	{
		int id;
		int ver;
		bool sharesRAM;
		bool isCPU;

		std::string simpleName;
		std::string name;
		std::string halfFpConfig;
		cl::Device device;
		Device(cl::Device dev = cl::Device(), int idPrm = -1, bool sharesRAMPrm = false, bool isCPUPrm = false);
	};
}

#endif // !GPGPU_DEVICE_LIB