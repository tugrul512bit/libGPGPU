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
	// an object for managing devices, kernels, worker cpu threads, load-balancing and creating/using parameters
	struct Computer
	{

		const static int DEVICE_ALL = 1 + 2 + 4;
		const static int DEVICE_GPUS = 1;
		const static int DEVICE_CPUS = 2;
		const static int DEVICE_ACCS = 4;
		const static int DEVICE_SELECTION_ALL = -1;

	private:
		std::map<std::string, std::vector<double>> loadBalances;
		std::vector<size_t> offsets;
		std::vector<size_t> ranges;
		std::map<std::string, std::vector<std::vector<double>>> oldLoadBalances;

		GPGPU_LIB::PlatformManager platform;
		std::vector<std::shared_ptr<GPGPU_LIB::Worker>> workers;
		std::map<std::string, GPGPU::HostParameter> hostParameters;
		std::mutex compileLock; // serialize device code compilations

		// kernel to parameters to position mapping
		std::map<std::string, std::map<std::string, int>> kernelParameters;
		/*
			deviceSelection = Computer::DEVICE_ALL ==> uses all gpu & cpu devices

		*/
	public:
		/*
		deviceSelection: selects type of devices to be queried. DEVICE_GPUS, DEVICE_CPUS, DEVICE_ACCS, DEVICE_ALL
		selectionIndex >= 0: selects single device from queried device list by index
		selectionIndex == -1 (DEVICE_SELECTION_ALL):  selects all devices from query list
		clonesPerDevice: number of times each physical device is duplicated in worker thread array to: overlap I/O to gain more performance, higher load-balancing quality
		CPU device is not cloned and is taken few of its threads to be dedicated for controling other devices fast. 
		If there are 4 GPU devices, then a 24-thread CPU is used as a 20-thread CPU by OpenCL's device fission feature and 4 threads serve the GPUs efficiently.
		*/
		Computer(int deviceSelection, int selectionIndex = DEVICE_SELECTION_ALL, int clonesPerDevice = 1);

		// returns number of queried devices (sum of devices from all platforms)
		int getNumDevices();

		/* compiles kernel code for given kernel name(that needs to be same as the function name in the kernel code) for all devices
		* not thread-safe between multiple Computer objects
		*/
		void compile(std::string kernelCode, std::string kernelName);

		/* 
		parameterName: parameter's name that is used when binding to kernel by setKernelParameter() or by method chaining ( computer.compute(  a.next(b).next(c), "kernelName",..   )  )
		numElements: number of elements with selected type (template parameter such as int, uint, int8_t, etc)
		numElementsPerThread: number of elements with selected type accessed by each work-item / gpu-thread / smallest work unit in OpenCL
				total number of global threads (work-items) to run = numElements / numElementsPerThread
		isInput = true ==> this parameter's host data is copied to devices before kernel is run (each device gets its own region unless isInputWithAllElements=true)
		isOutput=true ==> this parameter's devices' data are copied to host after kernel is run (each device copies its own regio)
		isInputWithAllElements=true ==> whole buffer is read instead of thread's own region when isInput=true. This is useful when all devices need a copy of whole array.
		!!! host parameter can only be input-only or output-only (currently) (because this lets all devices run independently without extra synchronization cost) !!!
		*/
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

		// applies load-balancing inside each call (better for uneven workloads per work-item)
		void runFineGrainedLoadBalancing(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize);

		// applies load-balancing between calls (better for even workloads per work-item)
		void run(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads);

		// works same as run with default parameters of fineGrainedLoadBalancing = false and fineGrainSize = 0
		// works same as runFineGrainedLoadBalancing with fineGrainedLoadBalancing = true (which sets fineGrainSize = numLocalThreads that may not be optimal for performance for too high global threads)
		void compute(
			GPGPU::HostParameter prm,
			std::string kernelName,
			size_t offsetElement,
			size_t numGlobalThreads,
			size_t numLocalThreads,
			bool fineGrainedLoadBalancing = false,
			size_t fineGrainSize = 0);

		// returns list of device names with their opencl version support
		std::vector<std::string> deviceNames();
	};
}
#endif // !GPGPU_COMPUTER_LIB