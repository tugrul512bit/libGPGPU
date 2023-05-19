#pragma once
#include "gpgpu_init.hpp"
#include "device.h"
namespace GPGPU
{
	struct PlatformManager
	{
		std::vector<cl::Platform> platforms;

		PlatformManager()
		{
			cl_int op = cl::Platform::get(&platforms);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("Platform::get error: ") + std::to_string(op));
			}
		}

		void printPlatforms()
		{
			for (auto& p : platforms)
			{
				std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
			}
		}


		std::vector<Device> getDevices(int typeOfDevice, int nOtherDevices = 0)
		{
			std::vector<Device> devices;
			int countId = 0;
			for (int i = 0; i < platforms.size(); i++)
			{
				std::vector<cl::Device> devicesTmp;
				cl_int op = platforms[i].getDevices(typeOfDevice, &devicesTmp);
				if (op != CL_SUCCESS)
				{
					throw std::invalid_argument(std::string("getDevices error: ") + std::to_string(op));
				}

				for (int j = 0; j < devicesTmp.size(); j++)
				{
					bool duplicate = false;
					for (int k = 0; k < devices.size(); k++)
					{
						if (devices[k].device.get() == devicesTmp[j].get())
						{
							duplicate = true;
							break;
						}
					}
					if (!duplicate)
					{
						cl_bool sharesRAM;
						bool isCPU = (CL_DEVICE_TYPE_CPU == typeOfDevice);
						cl_int op = devicesTmp[j].getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &sharesRAM);
						if (op != CL_SUCCESS)
						{
							throw std::invalid_argument(std::string("getInfo CL_DEVICE_HOST_UNIFIED_MEMORY error: ") + std::to_string(op));
						}

						// if there are other devices too, leave some threads for their control
						if ((nOtherDevices > 0) && (CL_DEVICE_TYPE_CPU == typeOfDevice))
						{
							// leaving some of threads for other devices' i/o control
							cl_device_partition_property p[]{ CL_DEVICE_PARTITION_BY_COUNTS, std::thread::hardware_concurrency() - nOtherDevices, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };
							std::vector<cl::Device> clDevices;
							cl_int err_create;
							if ((err_create = devicesTmp[j].createSubDevices(p, &clDevices)) == CL_SUCCESS)
							{
								devicesTmp[j] = clDevices[0];
							}
						}

						Device dev(devicesTmp[j], countId++, (CL_DEVICE_TYPE_CPU == typeOfDevice) || sharesRAM, isCPU);

						devices.push_back(dev);
					}
				}
			}
			return devices;
		}
	};
}