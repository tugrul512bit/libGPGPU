#include "worker.h"

namespace GPGPU_LIB
{

	Worker::Worker(Device dev) :working(true)
	{

		context = Context(dev);
		queue = CommandQueue(context);


		if (dev.id >= 0)
		{
			workerThread = std::thread([this]() {
				try {
					this->work();
				}
				catch (std::exception& ex)
				{
					std::cout << "error in worker thread: " << std::endl;
					std::cout << ex.what() << std::endl;
					exit(1);
				}
				});
		}
	}



	void Worker::work()
	{
		bool isWorking = true;
		size_t nanoLastCommand = 0;
		size_t workLastCommand = 0;
		while (isWorking)
		{


			GPGPUTask task = taskQueue.pop();

			switch (task.taskType)
			{
			case (GPGPUTask::GPGPU_TASK_COMPILE):
			{
				std::lock_guard<std::mutex> lg(*task.mutexPtr);
				mapKernelNameToKernel[task.kernelName] = Kernel(*task.conPtr, task.kernelCode, task.kernelName);
				break;
			}

			case (GPGPUTask::GPGPU_TASK_MIRROR):
			{

				mapParameterNameToParameter[task.hostParPtr->getName()] = Parameter(*task.conPtr, *task.hostParPtr);
				break;
			}

			case (GPGPUTask::GPGPU_TASK_STOP):
			{

				isWorking = false;
				break;
			}


			case (GPGPUTask::GPGPU_TASK_COMPUTE):
			{
				workLastCommand = 0;
				{
					GPGPU::Bench bench(&nanoLastCommand);
					Kernel& kernel = mapKernelNameToKernel[task.kernelName];
					task.comQuePtr->copyInputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);
					task.comQuePtr->run(kernel, task.globalOffset, task.globalSize, task.localSize, task.offset);
					task.comQuePtr->copyOutputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);
					workLastCommand += task.globalSize;

					task.comQuePtr->sync();
				}

				break;
			}


			case (GPGPUTask::GPGPU_TASK_COMPUTE_MULTIPLE):
			{
				workLastCommand = 0;
				{
					GPGPU::Bench bench(&nanoLastCommand);
					const int nK = task.kernelNames.size();
					for (int i = 0; i < nK; i++)
					{
						Kernel& kernel = mapKernelNameToKernel[task.kernelNames[i]];
						task.comQuePtr->copyInputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);
						task.comQuePtr->run(kernel, task.globalOffset, task.globalSize, task.localSize, task.offset);
						task.comQuePtr->copyOutputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);
						workLastCommand += task.globalSize;
					}
					task.comQuePtr->sync();
				}

				break;
			}

			case (GPGPUTask::GPGPU_TASK_COMPUTE_ALL):
			{
				workLastCommand = 0;
				{
					GPGPU::Bench bench(&nanoLastCommand);
					GPGPUTask taskNew;
					while ((taskNew = task.sharedTaskQueue->pop()).taskType != GPGPUTask::GPGPU_TASK_NULL)
					{
						Kernel& kernel = mapKernelNameToKernel[taskNew.kernelName];
						task.comQuePtr->copyInputsOfKernel(kernel, taskNew.globalOffset, taskNew.offset, taskNew.globalSize);
						task.comQuePtr->run(kernel, taskNew.globalOffset, taskNew.globalSize, taskNew.localSize, taskNew.offset);
						task.comQuePtr->copyOutputsOfKernel(kernel, taskNew.globalOffset, taskNew.offset, taskNew.globalSize);
						workLastCommand += taskNew.globalSize;
						task.comQuePtr->sync();

					}

				}

				break;
			}

			case (GPGPUTask::GPGPU_TASK_ARG):
			{

				Kernel& kernel = mapKernelNameToKernel[task.kernelName];
				Parameter& parameter = mapParameterNameToParameter[task.parameterName];
				task.comQuePtr->setPrm(kernel, parameter, task.parameterPosition);
				break;
			}

			default: break;
			}

			if (task.taskType != 0 || isWorking)
			{

				std::unique_lock<std::mutex> lock(commonSync);
				if (task.taskType == GPGPUTask::GPGPU_TASK_COMPUTE || task.taskType == GPGPUTask::GPGPU_TASK_COMPUTE_ALL)
				{
					benchmarks[task.kernelName] = nanoLastCommand;
					works[task.kernelName] = workLastCommand;
				}


				isWorking = working;
			}

			retireQueue.push(GPGPU_LIB::GPGPUTask());
			cond.notify_all();

			if (!isWorking)
			{

				break;
			}
		}

	}

	void Worker::stop()
	{
		bool workingTmp = true;
		{
			std::unique_lock<std::mutex> lock(commonSync);
			workingTmp = working;
			working = false;
		}

		if (workingTmp)
		{
			GPGPUTask task;
			task.taskType = GPGPUTask::GPGPU_TASK_STOP;
			taskQueue.push(task);
			waitAllTasks();
			workerThread.join();
		}
	}

	void Worker::runTasks(std::shared_ptr<GPGPUTaskQueue> taskQueueShared, std::string kernelName)
	{
		GPGPUTask task;
		task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE_ALL;
		task.sharedTaskQueue = taskQueueShared;
		task.comQuePtr = &queue;
		task.kernelName = kernelName;
		taskQueue.push(task);
	}

	void Worker::compile(std::string kernel, std::string kernelName, std::mutex* compileLock)
	{
		{
			std::unique_lock<std::mutex> lock(commonSync);
			benchmarks[kernelName] = 1;
		}
		GPGPUTask task;
		task.taskType = GPGPUTask::GPGPU_TASK_COMPILE;
		task.kernelCode = kernel;
		task.kernelName = kernelName;
		task.conPtr = &context;
		task.mutexPtr = compileLock;
		taskQueue.push(task);
		waitAllTasks();
	}

	void Worker::mirror(GPGPU::HostParameter* hostParameter)
	{
		GPGPUTask task;
		task.taskType = GPGPUTask::GPGPU_TASK_MIRROR;
		task.hostParPtr = hostParameter;
		task.conPtr = &context;
		taskQueue.push(task);
		waitAllTasks();
	}

	void Worker::setArg(std::string kernelName, std::string parameterName, int parameterIndex)
	{
		GPGPUTask task;
		task.taskType = GPGPUTask::GPGPU_TASK_ARG;
		task.kernelName = kernelName;
		task.parameterName = parameterName;
		task.parameterPosition = parameterIndex;
		task.comQuePtr = &queue;
		taskQueue.push(task);
		waitAllTasks();
	}

	void Worker::waitAllTasks()
	{
		retireQueue.pop();
	}

	void Worker::run(std::string kernelName, size_t globalOffset, size_t offset, size_t numGlobal, size_t numLocal, bool multipleKernels, std::vector<std::string> kernelNames)
	{
		GPGPUTask task;
		if (multipleKernels)
		{
			task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE_MULTIPLE;
			task.kernelNames = kernelNames;
			task.offset = offset;
			task.globalSize = numGlobal;
			task.localSize = numLocal;
			task.globalOffset = globalOffset;
			task.comQuePtr = &queue;
		}
		else
		{
			task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE;
			task.kernelName = kernelName;
			task.offset = offset;
			task.globalSize = numGlobal;
			task.localSize = numLocal;
			task.globalOffset = globalOffset;
			task.comQuePtr = &queue;
		}
		taskQueue.push(task);
	}

	std::string Worker::deviceName()
	{
		return context.device.name;
	}
	std::string Worker::deviceNameSimple()
	{
		return context.device.simpleName;
	}
	Worker::~Worker()
	{
		stop();
	}

}