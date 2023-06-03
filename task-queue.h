#pragma once

#ifndef GPGPU_TASK_QUEUE_LIB
#define GPGPU_TASK_QUEUE_LIB




#include "gpgpu_init.hpp"
#include "parameter.h"
#include "command-queue.h"
#include "context.h"
namespace GPGPU_LIB
{
	struct GPGPUTaskQueue;
	struct GPGPUTask
	{
		const static int GPGPU_TASK_NULL = 0;
		const static int GPGPU_TASK_COMPILE = 1;
		const static int GPGPU_TASK_ARG = 2;
		const static int GPGPU_TASK_MIRROR = 3;
		const static int GPGPU_TASK_COMPUTE = 4;
		const static int GPGPU_TASK_STOP = 5;
		const static int GPGPU_TASK_RETURN_NANO_BENCH = 6;
		const static int GPGPU_TASK_COMPUTE_ALL = 7;
		const static int GPGPU_TASK_COMPUTE_MULTIPLE = 8;
		std::string kernelCode;
		std::string kernelName;
		std::vector<std::string> kernelNames;
		std::string parameterName;
		int parameterPosition;
		size_t offset;
		size_t globalSize;
		size_t localSize;
		size_t globalOffset;
		GPGPU::HostParameter* hostParPtr;
		CommandQueue* comQuePtr;
		std::shared_ptr<GPGPUTaskQueue> sharedTaskQueue;
		Context* conPtr;
		std::mutex* mutexPtr;

		// no task = 0
		// compile a kernel = 1
		// bind argument to kernel = 2
		// mirror a host buffer on device memory (allcoate) = 3
		// compute a kernel (copy input + run kernel + copy output) = 4
		// stop working = 5
		// benchmark execution = 6 (for load-balancing)
		int taskType;


		GPGPUTask();

	};

	struct GPGPUTaskQueue
	{
		std::mutex syncPoint;
		std::condition_variable condition;
		std::queue<GPGPUTask> tasks;

		GPGPUTaskQueue();

		bool inProgress();

		void push(GPGPUTask task);

		GPGPUTask pop();
	};

}

#endif // !GPGPU_TASK_QUEUE_LIB