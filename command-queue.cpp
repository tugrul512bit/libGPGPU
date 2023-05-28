#pragma once

#include "command-queue.h"
namespace GPGPU_LIB
{
	CommandQueue::CommandQueue(Context con) :queue(con.context, con.device.device)
	{
		sharesRAM = con.device.sharesRAM;
	}

	void CommandQueue::run(Kernel& kernel, size_t globalOffset, size_t nGlobal, size_t nLocal, size_t offset)
	{
		cl_int op = queue.enqueueNDRangeKernel(kernel.kernel, cl::NDRange(offset + globalOffset), cl::NDRange(nGlobal), cl::NDRange(nLocal));
		if (op != CL_SUCCESS)
		{
			throw std::invalid_argument(std::string("enqueueNDRangeKernel error: ") + getErrorString(op));
		}
	}

	template<int T>
	struct ContainerT
	{
		int8_t data[T];
	};

	template<int T>
	constexpr void bigSwitchCase(Parameter& prm, Kernel& kernel, const int idx)
	{
		if (T == 1 && T == prm.hostPrm.getElementSize())
		{
			kernel.kernel.setArg(idx, prm.hostPrm.access<ContainerT<1>>(0));
			return;
		}



		if (T == prm.hostPrm.getElementSize())
		{
			kernel.kernel.setArg(idx, prm.hostPrm.access<ContainerT<T>>(0));
		}
		else if (T > 1)
		{
			bigSwitchCase<T - 1>(prm, kernel, idx);
		}

		return;
	}

	template<>
	void bigSwitchCase<1>(Parameter& prm, Kernel& kernel, const int idx)
	{
		if (1 == prm.hostPrm.getElementSize())
		{
			kernel.kernel.setArg(idx, prm.hostPrm.access<ContainerT<1>>(0));
			return;
		}
	}

	void CommandQueue::setPrm(Kernel& kernel, Parameter& prm, int idx)
	{
		cl_int op = 0;
		kernel.mapParameterNameToParameter[prm.name] = prm;
		if (prm.hostPrm.n == 1)
		{
			// scalar
			bigSwitchCase<256>(prm,kernel,idx);
			/*
			switch (prm.hostPrm.elementSize)
			{
			case 1:							
				kernel.kernel.setArg(idx, prm.hostPrm.access<int8_t[1]>(0));
				break;
			case 2:
				kernel.kernel.setArg(idx, prm.hostPrm.access<int8_t[2]>(0));
				break;
			case 3:
				kernel.kernel.setArg(idx, prm.hostPrm.access<int8_t[3]>(0));
				break;
			case 4:
				kernel.kernel.setArg(idx, prm.hostPrm.access<int8_t[4]>(0));
				break;
			default:break;
			}
			*/
		}
		else
		{
			// array
			op = kernel.kernel.setArg(idx, prm.buffer);
		}
		
		if (op != CL_SUCCESS)
		{
			throw std::invalid_argument(std::string("setArg error: ") + getErrorString(op));
		}
	}

	void CommandQueue::copyInputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement)
	{
		if (!sharesRAM)
		{
			for (auto& e : kernel.mapParameterNameToParameter)
			{

				if (e.second.readOp)
				{

					cl_int op = queue.enqueueWriteBuffer(
						e.second.buffer,
						CL_FALSE,
						e.second.readAll ? 0 : (globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread),
						e.second.readAll ? (e.second.elementSize * e.second.n) : (numElement * e.second.elementSize * e.second.elementsPerThread),
						e.second.hostPrm.quickPtr +
						(
							e.second.readAll ? 0 : (globalOffset * e.second.elementSize + offsetElement * e.second.elementSize * e.second.elementsPerThread)
							)
					);
					if (op != CL_SUCCESS)
					{
						throw std::invalid_argument(std::string("enqueueReadBuffer error: ") + getErrorString(op));
					}
				}
			}
		}
		else
		{
			for (auto& e : kernel.mapParameterNameToParameter)
			{

				if (e.second.readOp)
				{

					cl_int op;
					void* ptrMap = queue.enqueueMapBuffer(
						e.second.buffer,
						CL_FALSE,
						CL_MAP_WRITE,
						e.second.readAll ? 0 : (globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread),
						e.second.readAll ? (e.second.elementSize * e.second.n) : (numElement * e.second.elementSize * e.second.elementsPerThread),
						nullptr,
						nullptr,
						&op
					);

					if (op != CL_SUCCESS)
					{
						throw std::invalid_argument(std::string("enqueueMapBuffer(write) error: ") + getErrorString(op));
					}

					op = queue.enqueueUnmapMemObject(e.second.buffer, ptrMap, NULL, NULL);
					if (op != CL_SUCCESS)
					{
						throw std::invalid_argument(std::string("enqueueUnmapMemObject(write) error: ") + getErrorString(op));
					}
				}
			}

		}
	}

	void CommandQueue::copyOutputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement)
	{
		if (!sharesRAM)
		{
			for (auto& e : kernel.mapParameterNameToParameter)
			{

				if (e.second.writeOp)
				{

					cl_int op = queue.enqueueReadBuffer(
						e.second.buffer,
						CL_FALSE,
						(globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread),
						(numElement * e.second.elementSize * e.second.elementsPerThread),
						e.second.hostPrm.quickPtr +
						(
							(globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread)
							)
					);
					if (op != CL_SUCCESS)
					{
						std::string err1 = std::string("global offset = ") + std::to_string(globalOffset) + "\n";
						err1 += std::string("offset = ") + std::to_string(offsetElement) + "\n";
						err1 += std::string("num element = ") + std::to_string(numElement) + "\n";
						throw std::invalid_argument(std::string("enqueueWriteBuffer-1 error: ") + getErrorString(op)+err1);
					}
				}
			}
		}
		else
		{
			for (auto& e : kernel.mapParameterNameToParameter)
			{

				if (e.second.writeOp)
				{

					cl_int op;
					void* ptrMap = queue.enqueueMapBuffer(
						e.second.buffer,
						CL_FALSE,
						CL_MAP_READ,
						(globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread),
						(numElement * e.second.elementSize * e.second.elementsPerThread),
						nullptr,
						nullptr,
						&op
					);

					if (op != CL_SUCCESS)
					{
						throw std::invalid_argument(std::string("enqueueMapBuffer(read) error: ") + getErrorString(op));
					}

					op = queue.enqueueUnmapMemObject(e.second.buffer, ptrMap, NULL, NULL);
					if (op != CL_SUCCESS)
					{
						throw std::invalid_argument(std::string("enqueueUnmapMemObject(read) error: ") + getErrorString(op));
					}
				}
			}
		}
	}

	void CommandQueue::flush()
	{
		cl_int op = queue.flush();
		if (op != CL_SUCCESS)
		{
			throw std::invalid_argument(std::string("flush error: ") + getErrorString(op));
		}
	}

	void CommandQueue::sync()
	{
		cl_int op = queue.finish();
		if (op != CL_SUCCESS)
		{
			throw std::invalid_argument(std::string("finish error: ") + getErrorString(op));
		}
	}

}