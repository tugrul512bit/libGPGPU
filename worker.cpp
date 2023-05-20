#include "worker.h"

namespace GPGPU
{

		Worker::Worker(Device dev) :working(true), currentWorkComplete(true)
		{
			nano = 1;
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
			size_t nanoLastCommand = 1;
			while (isWorking)
			{
				nanoLastCommand = 1;

				GPGPUTask task = taskQueue.pop();

				if (task.taskType != 0)
				{

					std::unique_lock<std::mutex> lock(commonSync);
					currentWorkComplete = false;
				}

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

					mapParameterNameToParameter[task.hostParPtr->name] = Parameter(*task.conPtr, *task.hostParPtr);
					break;
				}

				case (GPGPUTask::GPGPU_TASK_STOP):
				{

					isWorking = false;
					break;
				}


				case (GPGPUTask::GPGPU_TASK_COMPUTE):
				{

					{
						Bench bench(&nanoLastCommand);
						Kernel& kernel = mapKernelNameToKernel[task.kernelName];
						task.comQuePtr->copyInputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);
						task.comQuePtr->run(kernel, task.globalOffset, task.globalSize, task.localSize, task.offset);
						task.comQuePtr->copyOutputsOfKernel(kernel, task.globalOffset, task.offset, task.globalSize);

						task.comQuePtr->sync();
					}

					break;
				}

				case (GPGPUTask::GPGPU_TASK_COMPUTE_ALL):
				{

					{
						Bench bench(&nanoLastCommand);
						GPGPUTask taskNew;
						while ((taskNew = task.sharedTaskQueue->pop()).taskType != GPGPUTask::GPGPU_TASK_NULL)
						{
							Kernel& kernel = mapKernelNameToKernel[taskNew.kernelName];
							task.comQuePtr->copyInputsOfKernel(kernel, taskNew.globalOffset, taskNew.offset, taskNew.globalSize);
							task.comQuePtr->run(kernel, taskNew.globalOffset, taskNew.globalSize, taskNew.localSize, taskNew.offset);
							task.comQuePtr->copyOutputsOfKernel(kernel, taskNew.globalOffset, taskNew.offset, taskNew.globalSize);
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
					if (task.taskType == GPGPUTask::GPGPU_TASK_COMPUTE)
						benchmarks[task.kernelName] = nanoLastCommand;
					this->nano = nanoLastCommand;
					currentWorkComplete = true;
					isWorking = working;
				}

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

		void Worker::runTasks(std::shared_ptr<GPGPUTaskQueue> taskQueueShared)
		{
			GPGPUTask task;
			task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE_ALL;
			task.sharedTaskQueue = taskQueueShared;
			task.comQuePtr = &queue;
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

		void Worker::mirror(HostParameter* hostParameter)
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
			bool wait = true;

			while (wait)
			{
				std::unique_lock<std::mutex> lock(commonSync);
				wait = taskQueue.inProgress() || !currentWorkComplete;
				if (wait)
				{
					cond.wait(lock, [&]() { return !(taskQueue.inProgress() || !currentWorkComplete); });
				}
			}
		}

		void Worker::run(std::string kernelName, size_t globalOffset, size_t offset, size_t numGlobal, size_t numLocal)
		{
			GPGPUTask task;
			task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE;
			task.kernelName = kernelName;
			task.offset = offset;
			task.globalSize = numGlobal;
			task.localSize = numLocal;
			task.globalOffset = globalOffset;
			task.comQuePtr = &queue;
			taskQueue.push(task);


		}

		std::string Worker::deviceName()
		{
			return context.device.name;
		}

		Worker::~Worker()
		{
			stop();
		}
	
}