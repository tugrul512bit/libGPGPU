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
		// if a buffer is meant to be read-write in kernel, then it can not be read/written from host side for optimization reasons so use it as read=false write=false that means only device can access it.
		if (read && write)
		{
			throw std::invalid_argument("Error: Buffer can not be both input and output at the same time. If kernel is meant to read/write this buffer arbitrarily, then use read=false write=false and access it within device freely as a state-management. This may also require an extra kernel to initialize the buffer.");
		}

		if (parameterName == "")
		{
			ptr = nullptr;
		}
		else
		{
			// allocate buffer with enough padding for alignment and size restrictions of mapping/unmapping of OpenCL buffer
			quickPtrVal = new int8_t[nElements * sizeElement + 4096 /* for re-alignment*/ + 4096 /* for zero-copy mapping CL_USE_HOST_PTR */];

			// align buffer
			size_t val = (size_t)quickPtrVal;
			while ((val % 4096) != 0)
			{
				val++;
			}

			ptr = std::shared_ptr<int8_t>(quickPtrVal, [](int8_t* pt) { if (pt) delete[] pt; }); // last host parameter standing releases memory
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
					 
					hostParameter.readOp ? 
						(CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY) : // host only writes, kernel only reads
						(hostParameter.writeOp?
							(CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY): // host only reads, kernel only writes
							CL_MEM_READ_WRITE  // meant for device-only usage like read+write from only kernel, not host
						)
					

				),

				hostParameter.elementSize * hostParameter.n,

				sharesRAM ? hostParameter.quickPtr : nullptr

			));
		}
}