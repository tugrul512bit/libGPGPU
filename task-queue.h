#pragma once
#include "gpgpu_init.hpp"
#include "parameter.h"
#include "command-queue.h"
#include "context.h"
namespace GPGPU
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
		std::string kernelCode;
		std::string kernelName;
		std::string parameterName;
		int parameterPosition;
		size_t offset;
		size_t globalSize;
		size_t localSize;
		size_t globalOffset;
		HostParameter* hostParPtr;
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


		GPGPUTask() :
			kernelCode(""),
			kernelName(""),
			parameterName(""),
			parameterPosition(0),
			offset(0),
			globalSize(0),
			localSize(0),
			hostParPtr(nullptr),
			comQuePtr(nullptr),
			taskType(0),
			conPtr(nullptr),
			mutexPtr(nullptr),
			sharedTaskQueue(nullptr),
			globalOffset(0)
		{}

	};

	struct GPGPUTaskQueue
	{
		std::mutex syncPoint;
		std::condition_variable condition;
		std::queue<GPGPUTask> tasks;

		GPGPUTaskQueue()
		{

		}

		bool inProgress()
		{
			std::lock_guard<std::mutex> lock(syncPoint);
			return tasks.size() > 0;
		}

		void push(GPGPUTask task)
		{
			std::lock_guard<std::mutex> lock(syncPoint);
			tasks.push(task);
			condition.notify_all();
		}

		GPGPUTask pop()
		{

			std::unique_lock<std::mutex> lock(syncPoint);

			while (tasks.size() == 0)
			{
				condition.wait(lock, [&]() { return tasks.size() > 0;  });
			}

			GPGPUTask result = tasks.front();
			tasks.pop();
			return result;
		}
	};

}