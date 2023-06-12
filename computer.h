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
		giveDirectRamAccessToCPU: OpenCL spec does not give permission to iGPU + CPU map/unmap on same host pointer simultaneously. So one has to pick iGPU or CPU to have direct-access (zero-copy) to RAM during computations.
			true = CPU gets direct RAM access
			false = iGPU gets direct RAM access
			the other one works same as a discrete device
		*/
		Computer(int deviceSelection, int selectionIndex = DEVICE_SELECTION_ALL, int clonesPerDevice = 1, bool giveDirectRamAccessToCPU=true);

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

		// creates a single item - array on host side but a scalar on device side
		template<typename T>
		HostParameter createScalarInput(std::string parameterName)
		{
			return createHostParameter<T>(parameterName, 1, 1, true, false, true);
		}

		// creates input array. All elements are copied to all devices.
		// use for randomly accessing any other data element within any work-item or device
		template<typename T>
		HostParameter createArrayInput(std::string parameterName, size_t numElements, size_t numElementsPerThread=1)
		{
			return createHostParameter<T>(parameterName, numElements, numElementsPerThread, true, false, true);
		}

		// creates input array. Devices get only their own elements.
		// use for embarrassingly-parallel data where neighboring data elements are not required
		template<typename T>
		HostParameter createArrayInputLoadBalanced(std::string parameterName, size_t numElements, size_t numElementsPerThread=1)
		{
			return createHostParameter<T>(parameterName, numElements, numElementsPerThread, true, false, false);
		}

		// creates output array. Devices copy only their own elements to the output because of possible race-conditions
		// works like createArrayInputLoadBalanced except for the output
		template<typename T>
		HostParameter createArrayOutput(std::string parameterName, size_t numElements, size_t numElementsPerThread=1)
		{
			return createHostParameter<T>(parameterName, numElements, numElementsPerThread, false, true, false);
		}

		// creates array that is not used for I/O with host (only meant for device-side state storage)
		template<typename T>
		HostParameter createArrayState(std::string parameterName, size_t numElements, size_t numElementsPerThread = 1)
		{
			return createHostParameter<T>(parameterName, numElements, numElementsPerThread, false, false, false);
		}

		// binds a parameter to a kernel at parameterPosition-th position
		void setKernelParameter(std::string kernelName, std::string parameterName, int parameterPosition);

		// applies load-balancing inside each call (better for uneven workloads per work-item)
		std::vector<double>  runFineGrainedLoadBalancing(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize);
		std::vector<double>  runFineGrainedLoadBalancingMultiple(std::vector<std::string> kernelNames, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads, size_t loadSize);

		/*
			applies load - balancing between calls(better for even workloads per work - item)
			returns workload ratios of devices (on the same order their names appear on deviceNames())
		*/
		std::vector<double> run(std::string kernelName, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads);
		std::vector<double> runMultiple(std::vector<std::string> kernelNames, size_t offsetElement, size_t numGlobalThreads, size_t numLocalThreads);

		// works same as run with default parameters of fineGrainedLoadBalancing = false and fineGrainSize = 0
		// works same as runFineGrainedLoadBalancing with fineGrainedLoadBalancing = true (which sets fineGrainSize = numLocalThreads that may not be optimal for performance for too high global threads)
		std::vector<double> compute(
			GPGPU::HostParameter prm,
			std::string kernelName,
			size_t offsetElement,
			size_t numGlobalThreads,
			size_t numLocalThreads,
			bool fineGrainedLoadBalancing = false,
			size_t fineGrainSize = 0);

		std::vector<double> computeMultiple(
			std::vector<GPGPU::HostParameter> prm,
			std::vector<std::string> kernelName,
			size_t offsetElement,
			size_t numGlobalThreads,
			size_t numLocalThreads,
			bool fineGrainedLoadBalancing = false,
			size_t fineGrainSize = 0);

		// returns list of device names with their opencl version support
		std::vector<std::string> deviceNames(bool detailed = true);
	};
}
#endif // !GPGPU_COMPUTER_LIB