#pragma once
#ifndef GPGPU_PARAMETER_LIB
#define GPGPU_PARAMETER_LIB


#include "gpgpu_init.hpp"
#include "context.h"

#include <memory>

namespace GPGPU
{
	struct Computer;
	// per-program allocated host memory
	struct HostParameter
	{
		std::string name;

		size_t n;
		size_t elementSize;
		size_t elementsPerThr;
		std::shared_ptr<int8_t> ptr;
		std::vector<std::string> prmList;
		// points to 4096-aligned region, has a size of multiple of 4096 bytes (for zero-copy access from CPU, iGPU)
		int8_t* quickPtr;
		int8_t* quickPtrVal;
		bool readOp;
		bool writeOp;
		bool readAllOp;
		HostParameter(
			std::string parameterName = "",
			size_t nElements = 1,
			size_t sizeElement = 1,
			size_t elementsPerThread = 1,
			bool read = false,
			bool write = false,
			bool readAll = false
		);

		// operator overloading from char buffer
		template<typename T>
		T& access(size_t index)
		{
			return *reinterpret_cast<T*>(quickPtr + (index * elementSize));
		}

		HostParameter next(HostParameter prm);

	
		
	};


	// per-device allocated memory
	struct Parameter
	{
		std::string name;
		size_t n;
		size_t elementSize;
		size_t elementsPerThread;
		cl::Buffer buffer;
		HostParameter hostPrm;
		bool readOp;
		bool writeOp;
		bool readAll;
		Parameter(Context con = Context(), HostParameter hostParameter = HostParameter());
	};


}

#endif // !GPGPU_PARAMETER_LIB