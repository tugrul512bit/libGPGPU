
#include "platform.h"
namespace GPGPU_LIB
{

		PlatformManager::PlatformManager()
		{
			std::vector<cl::Platform> tmp;
			cl_int op = cl::Platform::get(&tmp);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("Platform::get error: ") + getErrorString(op));
			}

			for (int i = 0; i < tmp.size(); i++)
			{
				int ver = tmp[i].getInfo<CL_PLATFORM_VERSION>(&op).at(7) - '0';
				if (op != CL_SUCCESS)
				{
					throw std::invalid_argument(std::string("Platformquery version error: ") + getErrorString(op));
				}
				if (ver >= 2 && CL_HPP_MINIMUM_OPENCL_VERSION >= 200)
				{
					platforms.push_back(tmp[i]);
				}
				else if (ver >= 1)
				{
					platforms.push_back(tmp[i]);
				}
			}
		}

		void PlatformManager::printPlatforms()
		{
			for (auto& p : platforms)
			{
				std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
			}
		}


		std::vector<Device> PlatformManager::getDevices(int typeOfDevice, int nOtherDevices)
		{
			std::vector<Device> devices;
			int countId = 0;
			for (int i = 0; i < platforms.size(); i++)
			{
				std::vector<cl::Device> devicesTmp;
				cl_int op2 = platforms[i].getDevices(typeOfDevice, &devicesTmp);
				if (op2 != CL_SUCCESS)
				{
					throw std::invalid_argument(std::string("getDevices error: ") + getErrorString(op2));
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
							throw std::invalid_argument(std::string("getInfo CL_DEVICE_HOST_UNIFIED_MEMORY error: ") + getErrorString(op));
						}
						// debugging
						//sharesRAM = false;
						// 
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

						Device dev(devicesTmp[j], countId++,  (CL_DEVICE_TYPE_CPU == typeOfDevice) || sharesRAM,  isCPU);

						devices.push_back(dev);
					}
				}
			}
			std::vector<Device> devicesResult;
			for (int i = 0; i < devices.size(); i++)
			{
				
				if (devices[i].ver >= CL_HPP_MINIMUM_OPENCL_VERSION)
				{
					devicesResult.push_back(devices[i]);
				}
			}
			return devicesResult;
		}

}