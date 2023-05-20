#pragma once
#ifndef GPGPU_DEVICE_LIB
#define GPGPU_DEVICE_LIB


#include "gpgpu_init.hpp"
#include<iostream>
namespace GPGPU
{
	struct Device
	{
		int id;
		int ver;
		bool sharesRAM;
		bool isCPU;

		std::string name;
		std::string halfFpConfig;
		cl::Device device;
		Device(cl::Device dev = cl::Device(), int idPrm = -1, bool sharesRAMPrm = false, bool isCPUPrm = false);
	};
}

#endif // !GPGPU_DEVICE_LIB