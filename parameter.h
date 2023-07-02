#pragma once
#ifndef GPGPU_PARAMETER_LIB
#define GPGPU_PARAMETER_LIB


#include "gpgpu_init.hpp"
#include "context.h"

#include <memory>
#include <algorithm>
// forward-declaring for friendship because only friends have access to private parts
namespace GPGPU_LIB
{
	struct Parameter;
	struct CommandQueue;
}

namespace GPGPU
{
	struct Computer;
	struct Worker;

	// per-program allocated host memory
	struct HostParameter
	{
		friend struct GPGPU_LIB::Parameter;
		friend struct Worker;
		friend struct GPGPU_LIB::CommandQueue;
		friend struct Computer;
	private:
		std::string name;
		size_t n;
		size_t elementSize;
		size_t elementsPerThr;
		std::shared_ptr<int8_t> ptr;
		std::vector<std::string> prmList;
		// points to 4096-aligned region, has a size of multiple of 4096 bytes (for zero-copy access from CPU, iGPU)
		// todo: also make the copies multiple of 4096 bytes (if that is the last chunk of parameter to copy [i.e. last device to run it] )
		int8_t* quickPtr;
		int8_t* quickPtrVal;
		bool readOp;
		bool writeOp;
		bool readAllOp;
	public:
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

		template<typename T>
		T* accessPtr(size_t index)
		{
			return reinterpret_cast<T*>(quickPtr + (index * elementSize));
		}

		HostParameter next(HostParameter prm);

		// read buffer and write to region starting at ptrPrm
		// numElements=0 means all elements are copied
		template<typename T>
		void copyDataToPtr(T * ptrPrm, size_t numElements=0, size_t elementOffset=0)
		{
			elementOffset = (numElements == 0 ? 0 : elementOffset);
			numElements = (numElements == 0 ? n : numElements);
			std::copy(
				reinterpret_cast<T*>(quickPtr + (elementOffset * elementSize)),
				reinterpret_cast<T*>(quickPtr + ((elementOffset + numElements) * elementSize)),
				ptrPrm);
		}

		// read region starting from ptrPrm and write to buffer
		// numElements=0 means all elements are copied
		template<typename T>
		void copyDataFromPtr(T* ptrPrm, size_t numElements=0, size_t elementOffset=0)
		{
			elementOffset = (numElements == 0 ? 0 : elementOffset);
			numElements = (numElements == 0 ? n : numElements);
			std::copy(
				ptrPrm,
				ptrPrm+numElements,
				reinterpret_cast<T*>(quickPtr + (elementOffset * elementSize))				
			);
		}

		std::string getName();		

		// number of bytes per element
		const size_t getElementSize() const
		{
			return elementSize;
		}

		// sets all elements to the newValue value
		template<typename T>
		void operator = (const T& newValue)
		{
			std::fill(
				reinterpret_cast<T*>(quickPtr),
				reinterpret_cast<T*>(quickPtr + (n * elementSize)),
				newValue
			);
		}

		void operator = (const HostParameter& hPrm)
		{
			elementSize = hPrm.elementSize;

			name=hPrm.name;
			n=hPrm.n;
			elementSize=hPrm.elementSize;
			elementsPerThr=hPrm.elementsPerThr;
			ptr=hPrm.ptr;
			prmList=hPrm.prmList;

			
			quickPtr=hPrm.quickPtr;
			quickPtrVal=hPrm.quickPtrVal;
			readOp=hPrm.readOp;
			writeOp=hPrm.writeOp;
			readAllOp=hPrm.readAllOp;
		}

	};
}

namespace GPGPU_LIB
{



	// per-device allocated memory
	struct Parameter
	{
		std::string name;
		size_t n;
		size_t elementSize;
		size_t elementsPerThread;
		cl::Buffer buffer;
		GPGPU::HostParameter hostPrm;
		bool readOp;
		bool writeOp;
		bool readAll;
		Parameter(Context con = Context(), GPGPU::HostParameter hostParameter = GPGPU::HostParameter());
	};


}

#endif // !GPGPU_PARAMETER_LIB