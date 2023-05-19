#pragma once
#include "gpgpu_init.hpp"
#include "device.h"
namespace GPGPU
{
	struct Context
	{
		cl::Context context;
		Device device;
		Context(Device dev = Device())
		{
			context = cl::Context(dev.device);
			device = dev;

		}

	};
}