#pragma once
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
		) :
			name(parameterName),
			n(nElements),
			elementSize(sizeElement),
			elementsPerThr(elementsPerThread),
			readOp(read),
			writeOp(write),
			readAllOp(readAll)
		{
			if (parameterName == "")
			{
				ptr = nullptr;
			}
			else
			{
				// align buffer
				quickPtrVal = new int8_t[nElements * sizeElement + 4096 /* for re-alignment*/ + 4096 /* for zero-copy mapping CL_USE_HOST_PTR */];
				size_t val = (size_t)quickPtrVal;

				while ((val % 4096) != 0)
				{
					val++;
				}

				ptr = std::shared_ptr<int8_t>(quickPtrVal, [](int8_t* pt) { delete[] pt; }); // last host parameter standing releases memory
				quickPtr = reinterpret_cast<int8_t*>(val);
			}
			prmList.push_back(parameterName);
		}

		// operator overloading from char buffer
		template<typename T>
		T& access(size_t index)
		{
			return *reinterpret_cast<T*>(quickPtr + (index * elementSize));
		}

		HostParameter next(HostParameter prm)
		{
			HostParameter result = *this;
			result.prmList.push_back(prm.name);
			return result;
		}

	
		
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
		Parameter(Context con = Context(), HostParameter hostParameter = HostParameter()) :
			name(hostParameter.name),
			n(hostParameter.n),
			elementSize(hostParameter.elementSize),
			hostPrm(hostParameter),
			readOp(hostParameter.readOp),
			writeOp(hostParameter.writeOp),
			readAll(hostParameter.readAllOp),
			elementsPerThread(hostParameter.elementsPerThr)
		{
			bool sharesRAM = con.device.sharesRAM;
			buffer = ((hostParameter.name == "") ? cl::Buffer() : cl::Buffer(con.context,

				(sharesRAM ? CL_MEM_USE_HOST_PTR : 0) |
				(
					(hostParameter.readOp && hostParameter.writeOp) ? CL_MEM_READ_WRITE : (hostParameter.readOp ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY)

					),

				hostParameter.elementSize * hostParameter.n,

				sharesRAM ? hostParameter.quickPtr : nullptr

			));
		}
	};


}