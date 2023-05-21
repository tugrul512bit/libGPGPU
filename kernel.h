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


namespace GPGPU_LIB
{
	// wrapper for  OpenCL kernel object with some helper fields to be re-used later
	struct Kernel
	{
		cl::Kernel kernel;
		std::string name;
		std::string code;
		Context context;
		bool isRunning; // todo: check this before setting an argument (and wait) and set this before running
		std::map<std::string, Parameter> mapParameterNameToParameter;

		/* compiles the given kernel code for the kernel name to be called later
		 todo: add caching for binary code, probably not needed if driver has its own caching
		 */
		Kernel(Context con = Context(), std::string kernelCode = "", std::string kernelName = "");
	};
}
#endif // !GPGPU_KERNEL_LIB