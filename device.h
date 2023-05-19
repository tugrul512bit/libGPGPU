#pragma once
#include "gpgpu_init.hpp"

namespace GPGPU
{
	struct Device
	{
		int id;
		bool sharesRAM;
		bool isCPU;
		cl::Device device;
		Device(cl::Device dev = cl::Device(), int idPrm = -1, bool sharesRAMPrm = false, bool isCPUPrm = false)
		{
			sharesRAM = sharesRAMPrm;
			device = dev;
			id = idPrm;
			isCPU = isCPUPrm;
		}
	};
}