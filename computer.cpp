#include "computer.h"

namespace GPGPU
{
	Computer::Computer(int deviceSelection, int selectionIndex, int clonesPerDevice, bool giveDirectRamAccessToCPU)
	{

		std::vector<GPGPU_LIB::Device> allGPUs = platform.getDevices(CL_DEVICE_TYPE_GPU);
		std::vector<GPGPU_LIB::Device> allACCs = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR);

		std::vector<GPGPU_LIB::Device> allDevices;



		if (deviceSelection & DEVICE_GPUS)
		{
			for (int i = 0; i < allGPUs.size(); i++)
			{
				// if this device shares RAM but user gives priority to CPU instead, then remove the direct-access feature from this device
				if (giveDirectRamAccessToCPU && allGPUs[i].sharesRAM)
				{
					allGPUs[i].sharesRAM = false;
				}
				allDevices.push_back(allGPUs[i]);
			}
		}

		if (deviceSelection & DEVICE_ACCS)
		{

			for (int i = 0; i < allACCs.size(); i++)
			{
				// if this device shares RAM but user gives priority to CPU instead, then remove the direct-access feature from this device
				if (giveDirectRamAccessToCPU && allACCs[i].sharesRAM)
				{
					allACCs[i].sharesRAM = false;
				}
				allDevices.push_back(allACCs[i]);
			}
		}

		const size_t nOtherDevices = allDevices.size();
		std::vector<GPGPU_LIB::Device> allCPUs = platform.getDevices(CL_DEVICE_TYPE_CPU, nOtherDevices);

		if (deviceSelection & DEVICE_CPUS)
		{
			for (int i = 0; i < allCPUs.size(); i++)
			{
				// if this CPU device shares RAM but user gives priority to GPU/ACC instead, then remove the direct-access feature from this CPU device
				if (!giveDirectRamAccessToCPU && allCPUs[i].sharesRAM)
				{
					allCPUs[i].sharesRAM = false;
				}
				allDevices.push_back(allCPUs[i]);
			}

		}




		int uniqueId = 0;
		bool cpuCloned = false;
		bool iGPUCloned = false;
		for (int j = 0; j < clonesPerDevice; j++)
		{
			std::vector<GPGPU_LIB::Device> selectedDevices;
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
				else if (selectedDevices[i].sharesRAM) // not a CPU, shares RAM = iGPU = not cloned either ()
				{
					if (iGPUCloned)
						ok = false;
					iGPUCloned = true;
				}

				if (ok)
				{

					offsets.push_back(1);
					ranges.push_back(1);
					selectedDevices[i].id = uniqueId++;// giving unique id to each device

					workers.push_back(std::make_shared<GPGPU_LIB::Worker>(selectedDevices[i]));
				}
			}
		}
	}

	int Computer::getNumDevices()
	{
		return workers.size();
	}

	void Computer::compile(std::string kernelCode, std::string kernelName)
	{
		for (int i = 0; i < workers.size(); i++)
		{
			workers[i]->compile(kernelCode, kernelName, &compileLock);
		}
	}



	// binds a parameter to a kernel at parameterPosition-th position
	void Computer::setKernelParameter(std::string kernelName, std::string parameterName, int parameterPosition)
	{
		// iterating 2 maps with several items should be faster than several threads to do something

		std::map<std::string, std::map<std::string, int>>::iterator it1 = kernelParameters.find(kernelName);
		bool sendToThreads = false;

		// if  kernel name not found (not run before)
		if (it1 == kernelParameters.end())
		{
			kernelParameters[kernelName][parameterName] = parameterPosition;
			sendToThreads = true;
		}
		else
		{
			std::map<std::string, int>::iterator it2 = it1->second.find(parameterName);

			// if parameter name not found in kernel settings (not used with this kernel before)
			if (it2 == it1->second.end())
			{
				it1->second[parameterName] = parameterPosition;
				sendToThreads = true;
			}
			else
			{
				// if parameter position changed, then update
				if (it2->second != parameterPosition)
				{
					it2->second = parameterPosition;
					sendToThreads = true;
				}

				// if parameter position did not change, do nothing
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
	std::vector<double> Computer::runFineGrainedLoadBalancing(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize)
	{
		std::vector<double> performancesOfDevices(workers.size());

		//std::cout << "debug 2.5" << std::endl;
		std::shared_ptr<GPGPU_LIB::GPGPUTaskQueue> taskQueue = std::make_shared<GPGPU_LIB::GPGPUTaskQueue>();

		//std::cout << "debug 4" << std::endl;
		for (size_t i = 0; i < numGlobalThreads; i += loadSize)
		{

			GPGPU_LIB::GPGPUTask task;
			task.taskType = GPGPU_LIB::GPGPUTask::GPGPU_TASK_COMPUTE;
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
			GPGPU_LIB::GPGPUTask task;
			task.taskType = GPGPU_LIB::GPGPUTask::GPGPU_TASK_NULL;
			taskQueue->push(task);
			workers[i]->runTasks(taskQueue,kernelName);
		}

		//std::cout << "debug 2" << std::endl;
		//std::cout << "debug 6" << std::endl;
		for (int i = 0; i < workers.size(); i++)
		{
			workers[i]->waitAllTasks();
		}
		//std::cout << "debug 3" << std::endl;

		double norm = 0.0;

		for (int i = 0; i < workers.size(); i++)
		{
			std::unique_lock<std::mutex> lock(workers[i]->commonSync);
	
			performancesOfDevices[i] = workers[i]->works[kernelName]/workers[i]->benchmarks[kernelName];
			norm += performancesOfDevices[i]; 
		}

		for (int i = 0; i < workers.size(); i++)
		{
			performancesOfDevices[i] /= norm; 
		}
		return performancesOfDevices;
	}


	// applies load-balancing between calls
	std::vector<double> Computer::run(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads)
	{
		const int n = workers.size();
		std::vector<double> nano(n);
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
			nano[i] = workers[i]->benchmarks[kernelName];

		}

		for (int i = 0; i < n; i++)
		{
			nano[i] = ranges[i] / nano[i]; // capability = run_size / run_time
			nano[i] = (avg[i] + (nano[i] * 4)) / (nlb + 4);
			totalLoad += nano[i];
		}

		// normalized loads
		for (int i = 0; i < n; i++)
		{
			avg[i] = nano[i];
			selectedKernelLB[i] = nano[i] / totalLoad;
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
			for (int i = 0; i < n; i++)
			{
				err += std::string("\n performance of device = ");
				err += std::to_string(workers[i]->benchmarks[kernelName]);
			}
			throw std::invalid_argument(err);
		}


		// compute offsets
		size_t curOfs = 0;
		for (int i = 0; i < n; i++)
		{
			offsets[i] = curOfs;
			curOfs += ranges[i];
		}

		// compute kernels with balanced loads
		for (int i = 0; i < n; i++)
		{

			workers[i]->run(kernelName, offsetElement, offsets[i], ranges[i], numLocalThreads);
		}


		// do some work while gpus are working independently
		oldLoadBalnc.push_back(avg);
		double norm = 0.0;
		
		for (int i = 0; i < n; i++)
		{
			nano[i] = ranges[i];
			norm += ranges[i];
		}

		for (int i = 0; i < n; i++)
		{
			nano[i] /= norm;
		}

		for (int i = 0; i < n; i++)
		{
			workers[i]->waitAllTasks();
		}

		return nano;
	}

	// applies load-balancing between calls
	std::vector<double> Computer::runMultiple(std::vector<std::string> kernelNames, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads)
	{
		std::string kernelName;
		for (auto& str : kernelNames)
		{
			kernelName += (str + " ");
		}
		const int n = workers.size();
		std::vector<double> nano(n);
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
			if(workers[i]->benchmarks.find(kernelName) == workers[i]->benchmarks.end())
				workers[i]->benchmarks[kernelName] = 1;
			nano[i] = workers[i]->benchmarks[kernelName];

		}

		for (int i = 0; i < n; i++)
		{
			nano[i] = ranges[i] / nano[i]; // capability = run_size / run_time
			nano[i] = (avg[i] + (nano[i] * 4)) / (nlb + 4);
			totalLoad += nano[i];
		}

		// normalized loads
		for (int i = 0; i < n; i++)
		{
			avg[i] = nano[i];
			selectedKernelLB[i] = nano[i] / totalLoad;
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
			for (int i = 0; i < n; i++)
			{
				err += std::string("\n performance of device = ");
				err += std::to_string(workers[i]->benchmarks[kernelName]);
			}
			throw std::invalid_argument(err);
		}


		// compute offsets
		size_t curOfs = 0;
		for (int i = 0; i < n; i++)
		{
			offsets[i] = curOfs;
			curOfs += ranges[i];
		}

		// compute kernels with balanced loads			
		for (int i = 0; i < n; i++)
		{
			workers[i]->run(kernelName, offsetElement, offsets[i], ranges[i], numLocalThreads, true, kernelNames);
		}
		

		// do some work while gpus are working independently
		oldLoadBalnc.push_back(avg);
		double norm = 0.0;

		for (int i = 0; i < n; i++)
		{
			nano[i] = ranges[i];
			norm += ranges[i];
		}

		for (int i = 0; i < n; i++)
		{
			nano[i] /= norm;
		}

		for (int i = 0; i < n; i++)
		{
			workers[i]->waitAllTasks();
		}
		
		return nano;
	}

	std::vector<double> Computer::compute(
		GPGPU::HostParameter prm,
		std::string kernelName,
		size_t offsetElement,
		size_t numGlobalThreads,
		size_t numLocalThreads,
		bool fineGrainedLoadBalancing,
		size_t fineGrainSize)
	{
		std::vector<double> performancesOfDevices;
		const int k = prm.prmList.size();
		for (int i = 0; i < k; i++)
		{
			setKernelParameter(kernelName, prm.prmList[i], i);
		}

		if (fineGrainedLoadBalancing)
			performancesOfDevices=runFineGrainedLoadBalancing(kernelName, offsetElement, numGlobalThreads, numLocalThreads, fineGrainSize == 0 ? numLocalThreads : fineGrainSize);
		else
			performancesOfDevices=run(kernelName, offsetElement, numGlobalThreads, numLocalThreads);
		return performancesOfDevices;
	}

	std::vector<double> Computer::computeMultiple(
		std::vector<GPGPU::HostParameter> prms,
		std::vector<std::string> kernelNames,
		size_t offsetElement,
		size_t numGlobalThreads,
		size_t numLocalThreads,
		bool fineGrainedLoadBalancing,
		size_t fineGrainSize)
	{
		std::map<std::string, bool> isSet;
		std::vector<double> performancesOfDevices;
		const int n = prms.size();

		for (int i = 0; i < n; i++)
		{
			auto kernIt = isSet.find(kernelNames[i]);
			if (kernIt == isSet.end())
			{
				const int k = prms[i].prmList.size();
				for (int j = 0; j < k; j++)
				{
					setKernelParameter(kernelNames[i], prms[i].prmList[j], j);
				}
				isSet.emplace(kernelNames[i], true);
			}
		}

		if (fineGrainedLoadBalancing)
		{
			const int nw = workers.size();
			performancesOfDevices.resize(nw);
			for (int j = 0; j < nw; j++)
			{
				performancesOfDevices[j] = 0;
			}
			for (int i = 0; i < n; i++)
			{
				auto performancesOfDevicesTmp = runFineGrainedLoadBalancing(kernelNames[i], offsetElement, numGlobalThreads, numLocalThreads, fineGrainSize == 0 ? numLocalThreads : fineGrainSize);
				for (int j = 0; j < nw; j++)
					performancesOfDevices[j] += performancesOfDevicesTmp[j];
			}
			for (int j = 0; j < nw; j++)
				performancesOfDevices[j] /= nw;
		}
		else
		{
			performancesOfDevices = runMultiple(kernelNames, offsetElement, numGlobalThreads, numLocalThreads);
		}
				
		return performancesOfDevices;
	}

	std::vector<std::string> Computer::deviceNames(bool detailed)
	{
		std::vector<std::string> names;
		for (int i = 0; i < workers.size(); i++)
		{
			if(detailed)
				names.push_back(std::string("Device ")+std::to_string(i)+std::string(": ") + workers[i]->deviceName() + (!workers[i]->context.device.sharesRAM ? " [direct-RAM-access disabled]" : ""));
			else
				names.push_back(workers[i]->deviceNameSimple());
		}
		return names;
	}
}