#pragma once


#ifndef GPGPU_COMMAND_QUEUE_LIB 
#define GPGPU_COMMAND_QUEUE_LIB


#include "gpgpu_init.hpp"
#include "context.h"
#include "device.h"
#include "parameter.h"
#include "kernel.h"

namespace GPGPU_LIB
{
	struct CommandQueue
	{
		cl::CommandQueue queue;
		bool sharesRAM;
		CommandQueue(Context con = Context());

		void run(Kernel& kernel, size_t globalOffset, size_t nGlobal, size_t nLocal, size_t offset);

		void copyFromParameter(Parameter& prm, const size_t nElements, const size_t startIndex);

		void copyToParameter(Parameter& prm, const size_t nElements, const size_t startIndex);


		void setPrm(Kernel& kernel, Parameter& prm, int idx);

		void copyInputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement);

		void copyOutputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement);

		void flush();

		void sync();
	};
}

#endif 