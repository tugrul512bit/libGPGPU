#pragma once
#ifndef GPGPU_COMPUTER_LIB
#define GPGPU_COMPUTER_LIB



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
		Computer(int deviceSelection, int selectionIndex = DEVICE_SELECTION_ALL, int clonesPerDevice = 1);

		int getNumDevices();

		void compile(std::string kernelCode, std::string kernelName);

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
		void setKernelParameter(std::string kernelName, std::string parameterName, int parameterPosition);

		// applies load-balancing inside each call
		void runFineGrainedLoadBalancing(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize);

		// applies load-balancing between calls
		void run(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads);

		void compute(
			HostParameter prm,
			std::string kernelName,
			size_t offsetElement,
			size_t numGlobalThreads,
			size_t numLocalThreads,
			bool fineGrainedLoadBalancing = false,
			size_t fineGrainSize = 0);

		std::vector<std::string> deviceNames();
	};
}
#endif // !GPGPU_COMPUTER_LIB