#include "task-queue.h"


namespace GPGPU_LIB
{
	struct GPGPUTaskQueue;


	GPGPUTask::GPGPUTask() :
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


		GPGPUTaskQueue::GPGPUTaskQueue()
		{

		}

		bool GPGPUTaskQueue::inProgress()
		{
			std::lock_guard<std::mutex> lock(syncPoint);
			return tasks.size() > 0;
		}

		void GPGPUTaskQueue::push(GPGPUTask task)
		{
			std::lock_guard<std::mutex> lock(syncPoint);
			tasks.push(task);
			condition.notify_all();
		}

		GPGPUTask GPGPUTaskQueue::pop()
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

}