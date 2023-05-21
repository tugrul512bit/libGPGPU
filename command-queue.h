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
	// opencl command queue wrapper that offers basic functionality: kernel execution, buffer copies. Setting parameter does not enqueue, it is an immediate operation but not thread-safe when kernel is already in use.
	struct CommandQueue
	{
		cl::CommandQueue queue;
		bool sharesRAM;
		// requires a context to build
		CommandQueue(Context con = Context());

		// runs a kernel with globalOffset starting thread offset, nGlobal number of global threads, nLocal number of local threads, offset thread offset that is unique to current device
		void run(Kernel& kernel, size_t globalOffset, size_t nGlobal, size_t nLocal, size_t offset);

		// sets a parameter for kernel with position idx that is zero-based
		void setPrm(Kernel& kernel, Parameter& prm, int idx);

		// copies (or no-copies for RAM-sharing devices) input buffers of kernel to devices from RAM
		void copyInputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement);

		// copies (or no-copies for RAM-sharing devices) output buffers of kernel from devices to RAM
		void copyOutputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement);

		// starts pushing commands to device
		void flush();

		// waits for device to complete all commands on current queue
		void sync();
	};
}

#endif 