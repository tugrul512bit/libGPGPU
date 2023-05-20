#pragma once
#ifndef GPGPU_KERNEL_LIB
#define GPGPU_KERNEL_LIB


#include <string>
#include <iostream>
#include <map>
#include "gpgpu_init.hpp"
#include "context.h"
#include "device.h"
#include "parameter.h"


namespace GPGPU
{
	struct Kernel
	{
		cl::Kernel kernel;
		std::string name;
		std::string code;
		Context context;
		bool isRunning;
		std::map<std::string, Parameter> mapParameterNameToParameter;
		Kernel(Context con = Context(), std::string kernelCode = "", std::string kernelName = "");
	};
}
#endif // !GPGPU_KERNEL_LIB