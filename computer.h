#pragma once
#include "gpgpu_init.hpp"
#include "worker.h"
#include "platform.h"
#include <map>
#include <memory>
#include <vector>
namespace GPGPU
{
	struct Computer
	{
		const static int DEVICE_ALL = 1 + 2 + 4;
		const static int DEVICE_GPUS = 1;
		const static int DEVICE_CPUS = 2;
		const static int DEVICE_ACCS = 4;
		const static int DEVICE_SELECTION_ALL = -1;
		std::map<std::string, std::vector<double>> loadBalances;
		std::vector<size_t> offsets;
		std::vector<size_t> ranges;
		std::map<std::string, std::vector<std::vector<double>>> oldLoadBalances;

		PlatformManager platform;
		std::vector<std::shared_ptr<Worker>> workers;
		std::map<std::string, HostParameter> hostParameters;
		std::mutex compileLock; // serialize device code compilations

		// kernel to parameters to position mapping
		std::map<std::string, std::map<std::string, int>> kernelParameters;
		/*
			deviceSelection = Computer::DEVICE_ALL ==> uses all gpu & cpu devices

		*/
		Computer(int deviceSelection, int selectionIndex = DEVICE_SELECTION_ALL, int clonesPerDevice = 1)
		{


			std::vector<Device> allGPUs = platform.getDevices(CL_DEVICE_TYPE_GPU);
			std::vector<Device> allACCs = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR);

			std::vector<Device> allDevices;



			if (deviceSelection & DEVICE_GPUS)
			{
				for (int i = 0; i < allGPUs.size(); i++)
					allDevices.push_back(allGPUs[i]);
			}

			if (deviceSelection & DEVICE_ACCS)
			{

				for (int i = 0; i < allACCs.size(); i++)
					allDevices.push_back(allACCs[i]);
			}

			const int nOtherDevices = allDevices.size();
			std::vector<Device> allCPUs = platform.getDevices(CL_DEVICE_TYPE_CPU, nOtherDevices);

			if (deviceSelection & DEVICE_CPUS)
			{
				for (int i = 0; i < allCPUs.size(); i++)
				{
					allDevices.push_back(allCPUs[i]);
				}

			}




			int uniqueId = 0;
			bool cpuCloned = false;
			for (int j = 0; j < clonesPerDevice; j++)
			{
				std::vector<Device> selectedDevices;
				if (selectionIndex == DEVICE_SELECTION_ALL)
				{
					selectedDevices = allDevices;
				}
				else
				{
					selectedDevices.push_back(allDevices[selectionIndex]);
				}

				for (int i = 0; i < selectedDevices.size(); i++)
				{
					bool ok = true;
					if (selectedDevices[i].isCPU)
					{
						if (cpuCloned)
							ok = false;

						cpuCloned = true;
					}
					if (ok)
					{

						offsets.push_back(1);
						ranges.push_back(1);
						selectedDevices[i].id = uniqueId++;// giving unique id to each device

						workers.push_back(std::make_shared<Worker>(selectedDevices[i]));
					}
				}
			}
		}

		int getNumDevices()
		{
			return workers.size();
		}

		void compile(std::string kernelCode, std::string kernelName)
		{
			for (int i = 0; i < workers.size(); i++)
			{
				workers[i]->compile(kernelCode, kernelName, &compileLock);
			}
		}

		// isInput=true ==> makes this an input parameter of kernels when attached to them (each thread reads its own region)
		// isOutput=true ==> makes this an output parameter of kernels when attached to them (each thread writes its own region)
		// isInputWithAllElements=true ==> whole buffer is read instead of thread's own region
		template<typename T>
		HostParameter createHostParameter(std::string parameterName, size_t numElements, size_t numElementsPerThread, bool isInput, bool isOutput, bool isInputWithAllElements)
		{
			hostParameters[parameterName] = HostParameter(parameterName, numElements, sizeof(T), numElementsPerThread, isInput, isOutput, isInputWithAllElements);
			for (int i = 0; i < workers.size(); i++)
			{
				workers[i]->mirror(&hostParameters[parameterName]);
			}
			return hostParameters[parameterName];
		}

		// binds a parameter to a kernel at parameterPosition-th position
		void setKernelParameter(std::string kernelName, std::string parameterName, int parameterPosition)
		{
			// iterating 2 maps with several items should be faster than several threads to do something

			std::map<std::string, std::map<std::string, int>>::iterator it1 = kernelParameters.find(kernelName);
			bool sendToThreads = false;
			if (it1 == kernelParameters.end())
			{
				kernelParameters[kernelName][parameterName] = parameterPosition;
				sendToThreads = true;
			}
			else
			{
				std::map<std::string, int>::iterator it2 = it1->second.find(parameterName);
				if (it2 == it1->second.end())
				{
					it1->second[parameterName] = parameterPosition;
					sendToThreads = true;
				}
				else
				{
					// if parameter position did not change, do nothing, if all else, do something
					if (it2->second != parameterPosition)
					{
						it2->second = parameterPosition;
						sendToThreads = true;
					}
				}
			}

			if (sendToThreads)
			{
				const int nWork = workers.size();
				for (int i = 0; i < nWork; i++)
				{
					workers[i]->setArg(kernelName, parameterName, parameterPosition);
				}
			}


		}

		// applies load-balancing inside each call
		void runFineGrainedLoadBalancing(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize)
		{
			//std::cout << "debug 2.5" << std::endl;
			std::shared_ptr<GPGPUTaskQueue> taskQueue = std::make_shared<GPGPUTaskQueue>();

			//std::cout << "debug 4" << std::endl;
			for (size_t i = 0; i < numGlobalThreads; i += loadSize)
			{

				GPGPUTask task;
				task.taskType = GPGPUTask::GPGPU_TASK_COMPUTE;
				task.kernelName = kernelName;
				task.globalOffset = offsetElement;
				task.offset = i;
				task.globalSize = loadSize;
				task.localSize = numLocalThreads;
				taskQueue->push(task);

			}
			//std::cout << "debug 5" << std::endl;
			// compute kernels with balanced loads
			//std::cout << "debug 1" << std::endl;
			for (int i = 0; i < workers.size(); i++)
			{
				// mark end of queue for each worker
				GPGPUTask task;
				task.taskType = GPGPUTask::GPGPU_TASK_NULL;
				taskQueue->push(task);
				workers[i]->runTasks(taskQueue);
			}
			
			//std::cout << "debug 2" << std::endl;
			//std::cout << "debug 6" << std::endl;
			for (int i = 0; i < workers.size(); i++)
			{
				workers[i]->waitAllTasks();
			}
			//std::cout << "debug 3" << std::endl;
			
		}

		// applies load-balancing between calls
		void run(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads)
		{
			const int n = workers.size();
			if (loadBalances.find(kernelName) == loadBalances.end())
			{
				loadBalances[kernelName] = std::vector<double>(n, 1.0);
			}

			std::vector<double>& selectedKernelLB = loadBalances[kernelName];


			// compute load-balancing

			auto& oldLoadBalnc = oldLoadBalances[kernelName];
			const int nlb = oldLoadBalnc.size();
			double totalLoad = 0;
			std::vector<double> avg(n, 0);
			for (int i = 0; i < nlb; i++)
			{
				for (int j = 0; j < n; j++)
				{
					avg[j] += oldLoadBalnc[i][j];
				}

			}
			for (int i = 0; i < n; i++)
			{
				std::unique_lock<std::mutex> lock(workers[i]->commonSync);
				workers[i]->nano = workers[i]->benchmarks[kernelName];
				workers[i]->nano = ranges[i] / workers[i]->nano; // capability = run_size / run_time
				workers[i]->nano = (avg[i] + (workers[i]->nano * 4)) / (nlb + 4);
				totalLoad += workers[i]->nano;
			}





			// normalized loads
			for (int i = 0; i < n; i++)
			{
				avg[i] = workers[i]->nano;
				selectedKernelLB[i] = workers[i]->nano / totalLoad;
			}




			for (int i = 0; i < nlb - 1; i++)
			{
				oldLoadBalnc[i] = oldLoadBalnc[i + 1];
			}

			if (oldLoadBalnc.size() > 1)
				oldLoadBalnc.resize(1);

			// calculate ranges
			for (int i = 0; i < n; i++)
			{
				ranges[i] = (((size_t)(numGlobalThreads * selectedKernelLB[i])) / numLocalThreads) * numLocalThreads;
			}

			size_t totalThreads = 0;

			// refine ranges (invalid values)
			for (int i = 0; i < n; i++)
			{
				// if no work was given, give it at least single work group 
				if (ranges[i] == 0)
				{
					ranges[i] = numLocalThreads;
				}
				totalThreads += ranges[i];
			}



			size_t toBeSubtracted = 0;
			size_t toBeAdded = 0;
			// can not compute more threads than given
			if (totalThreads > numGlobalThreads)
			{
				toBeSubtracted = totalThreads - numGlobalThreads;
			}

			// can not compute more threads than given
			if (totalThreads < numGlobalThreads)
			{
				toBeAdded = numGlobalThreads - totalThreads;
			}

			// redistribute the remainder
			// first devices tend to trade more of it, negligible when numGlobalThreads >> numLocalThreads
			int tryCt = 0;
			size_t newTotal = 0;
			while (toBeSubtracted > 0 || toBeAdded > 0)
			{
				if (tryCt++ > 10)
					break;
				for (int i = 0; i < n; i++)
				{
					if (toBeSubtracted > 0 && ranges[i] > numLocalThreads)
					{
						ranges[i] -= numLocalThreads;
						toBeSubtracted -= numLocalThreads;
					}

					if (toBeAdded > 0)
					{
						ranges[i] += numLocalThreads;
						toBeAdded -= numLocalThreads;
					}
				}
			}
			for (int i = 0; i < n; i++)
				newTotal += ranges[i];

			if (toBeSubtracted > 0 || toBeAdded > 0 || newTotal != numGlobalThreads || (newTotal / numLocalThreads) * numLocalThreads != newTotal)
			{
				std::string err("error: load-balancing failed. check if there is enough number of local threads in global number of threads to share between all devices and global must be integer-multiple of local");
				err += std::string(" \n  threads need to be subtracted =  ") + std::to_string(toBeSubtracted);
				err += std::string(" \n  threads need to be added =  ") + std::to_string(toBeAdded);
				err += std::string(" \n  total threads distributed =  ") + std::to_string(newTotal);
				err += std::string(" \n  global threads required =  ") + std::to_string(numGlobalThreads);
				throw std::invalid_argument(err);
			}



			// compute offsets
			size_t curOfs = 0;
			for (int i = 0; i < n; i++)
			{
				offsets[i] = curOfs;
				curOfs += ranges[i];
			}

			/*
			for (int i = 0; i < n; i++)
				std::cout<<
				workers[i]->context.device.device.getInfo<CL_DEVICE_NAME>() << ":        " << ranges[i] << "     isCPU:" <<
				workers[i]->context.device.isCPU << ":        " <<
				std::endl;
			*/


			// compute kernels with balanced loads
			for (int i = 0; i < n; i++)
			{

				workers[i]->run(kernelName, offsetElement, offsets[i], ranges[i], numLocalThreads);
			}

			oldLoadBalnc.push_back(avg);

			for (int i = 0; i < n; i++)
			{
				workers[i]->waitAllTasks();
			}
		}

		void compute(
			HostParameter prm, 
			std::string kernelName,
			size_t offsetElement, 
			size_t numGlobalThreads, 
			size_t numLocalThreads, 
			bool fineGrainedLoadBalancing = false,
			size_t fineGrainSize=0)
		{
			
			const int k = prm.prmList.size();
			for (int i = 0; i < k; i++)
			{
				setKernelParameter(kernelName, prm.prmList[i], i);
			}
		
			if (fineGrainedLoadBalancing)
				runFineGrainedLoadBalancing(kernelName, offsetElement, numGlobalThreads, numLocalThreads, fineGrainSize == 0 ? numLocalThreads : fineGrainSize);
			else
				run(kernelName, offsetElement, numGlobalThreads, numLocalThreads);
	
		}
	};
}