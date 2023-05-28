#include "device.h"
namespace GPGPU_LIB
{
	Device::Device(cl::Device dev, int idPrm, bool sharesRAMPrm, bool isCPUPrm )
	{
		sharesRAM = sharesRAMPrm;
		device = dev;
		id = idPrm;
		isCPU = isCPUPrm;
		cl_int op;
		if (id != -1)
		{
			name = device.getInfo<CL_DEVICE_NAME>(&op);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("error: device name query") + getErrorString(op));
			}

			// trim whitespace
			name.erase(name.begin(), std::find_if(name.begin(), name.end(), [](unsigned char ch) {
				return !std::isspace(ch);
				}));
			name.erase(std::find_if(name.rbegin(), name.rend(), [](unsigned char ch) {
				return !std::isspace(ch);
				}).base(), name.end());

			simpleName = name;
			name += " (";
			name += device.getInfo<CL_DEVICE_VERSION>(&op);
			name += " )";
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("error: device opencl version query") + getErrorString(op));
			}

			if (sharesRAM)
			{
				name += "[has direct access to RAM]";
			}

			std::string clVer = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>(&op);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("error: device opencl-c version query") + getErrorString(op));
			}
			if (clVer.size() < 10)
				ver = 120;
			else
			{
				if (clVer[9] >= '3')
					ver = 300;
				else if (clVer[9] >= '2')
					ver = 200;
				else
					ver = 120;
			}

		}
		else
		{
			name = "";
		}

		
	}
}