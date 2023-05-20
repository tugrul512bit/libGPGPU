#pragma once
#ifndef GPGPU_CONTEXT_LIB
#define GPGPU_CONTEXT_LIB



#include "gpgpu_init.hpp"
#include "device.h"
namespace GPGPU
{
	struct Context
	{
		cl::Context context;
		Device device;
		Context(Device dev = Device());

	};
}

#endif // !GPGPU_CONTEXT_LIB