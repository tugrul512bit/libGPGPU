#include "parameter.h"

namespace GPGPU
{
	struct Computer;

	HostParameter::HostParameter(
		std::string parameterName,
		size_t nElements,
		size_t sizeElement,
		size_t elementsPerThread,
		bool read,
		bool write,
		bool readAll
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

	HostParameter HostParameter::next(HostParameter prm)
	{
		HostParameter result = *this;
		result.prmList.push_back(prm.name);
		return result;
	}

	std::string HostParameter::getName()
	{
		return name;
	}
}

namespace GPGPU_LIB
{


		Parameter::Parameter(Context con, GPGPU::HostParameter hostParameter ) :
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
}