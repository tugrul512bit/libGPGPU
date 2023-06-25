#pragma once
#ifndef GPGPU_WORKER_LIB
#define GPGPU_WORKER_LIB

#include "gpgpu_init.hpp"
#include "device.h"
#include "context.h"
#include "parameter.h"
#include "kernel.h"
#include "command-queue.h"
#include "task-queue.h"
#include <map>
namespace GPGPU_LIB
{

	struct Worker
	{
		std::mutex commonSync;
		std::condition_variable cond;
		Context context;
		CommandQueue queue;
		std::map<std::string, Kernel> mapKernelNameToKernel;
		std::map<std::string, Parameter> mapParameterNameToParameter;
		GPGPUTaskQueue taskQueue;
		GPGPUTaskQueue retireQueue;
		bool working;


		std::map<std::string, double> benchmarks;
		std::map<std::string, size_t> works;
		std::thread workerThread;
		Worker(Device dev);

		void work();

		void stop();

		void runTasks(std::shared_ptr<GPGPUTaskQueue> taskQueueShared, std::string kernelName);

		void compile(std::string kernel, std::string kernelName, std::mutex* compileLock);

		void mirror(GPGPU::HostParameter* hostParameter);

		void setArg(std::string kernelName, std::string parameterName, int parameterIndex);

		void waitAllTasks();

		void run(std::string kernelName, size_t globalOffset, size_t offset, size_t numGlobal, size_t numLocal, bool multipleKernels = false, std::vector<std::string> kernelNames = std::vector<std::string>());

		std::string deviceName();
		std::string deviceNameSimple();
		~Worker();
	};
}

#endif // !GPGPU_WORKER_LIB