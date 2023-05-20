#pragma once
#ifndef GPGPU_PLATFORM_LIB
#define GPGPU_PLATFORM_LIB

#include "gpgpu_init.hpp"
#include "device.h"
#include <iostream>
namespace GPGPU_LIB
{
	struct PlatformManager
	{
		std::vector<cl::Platform> platforms;

		PlatformManager();

		void printPlatforms();


		std::vector<Device> getDevices(int typeOfDevice, int nOtherDevices = 0);
	};
}

#endif // !GPGPU_PLATFORM_LIB