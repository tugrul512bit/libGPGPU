#pragma once
#include "gpgpu_init.hpp"
#include "context.h"
#include "device.h"
#include "parameter.h"
#include "kernel.h"

namespace GPGPU
{
	struct CommandQueue
	{
		cl::CommandQueue queue;
		bool sharesRAM;
		CommandQueue(Context con = Context()) :queue(con.context, con.device.device)
		{
			sharesRAM = con.device.sharesRAM;
		}

		void run(Kernel& kernel, size_t globalOffset, size_t nGlobal, size_t nLocal, size_t offset)
		{
			cl_int op = queue.enqueueNDRangeKernel(kernel.kernel, cl::NDRange(offset + globalOffset), cl::NDRange(nGlobal), cl::NDRange(nLocal));
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("enqueueNDRangeKernel error: ") + std::to_string(op));
			}
		}

		void copyFromParameter(Parameter& prm, const size_t nElements, const size_t startIndex)
		{
			cl_int op = queue.enqueueReadBuffer(prm.buffer, CL_FALSE, startIndex * prm.elementSize, nElements * prm.elementSize, prm.hostPrm.quickPtr);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("enqueueReadBuffer error: ") + std::to_string(op));
			}
		}

		void copyToParameter(Parameter& prm, const size_t nElements, const size_t startIndex)
		{
			cl_int op = queue.enqueueWriteBuffer(prm.buffer, CL_FALSE, startIndex * prm.elementSize, nElements * prm.elementSize, prm.hostPrm.quickPtr);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("enqueueWriteBuffer-2 error: ") + std::to_string(op));
			}
		}


		void setPrm(Kernel& kernel, Parameter& prm, int idx)
		{
			kernel.mapParameterNameToParameter[prm.name] = prm;

			cl_int op = kernel.kernel.setArg(idx, prm.buffer);
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("setArg error: ") + std::to_string(op));
			}
		}

		void copyInputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement)
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
							throw std::invalid_argument(std::string("enqueueReadBuffer error: ") + std::to_string(op));
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
							throw std::invalid_argument(std::string("enqueueMapBuffer(write) error: ") + std::to_string(op));
						}

						op = queue.enqueueUnmapMemObject(e.second.buffer, ptrMap, NULL, NULL);
						if (op != CL_SUCCESS)
						{
							throw std::invalid_argument(std::string("enqueueUnmapMemObject(write) error: ") + std::to_string(op));
						}
					}
				}

			}
		}

		void copyOutputsOfKernel(Kernel& kernel, size_t globalOffset, size_t offsetElement, size_t numElement)
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
							throw std::invalid_argument(std::string("enqueueWriteBuffer-1 error: ") + std::to_string(op));
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
							CL_MAP_READ,
							(globalOffset * e.second.elementSize * e.second.elementsPerThread + offsetElement * e.second.elementSize * e.second.elementsPerThread),
							(numElement * e.second.elementSize * e.second.elementsPerThread),
							nullptr,
							nullptr,
							&op
						);

						if (op != CL_SUCCESS)
						{
							throw std::invalid_argument(std::string("enqueueMapBuffer(read) error: ") + std::to_string(op));
						}

						op = queue.enqueueUnmapMemObject(e.second.buffer, ptrMap, NULL, NULL);
						if (op != CL_SUCCESS)
						{
							throw std::invalid_argument(std::string("enqueueUnmapMemObject(read) error: ") + std::to_string(op));
						}
					}
				}
			}
		}

		void flush()
		{
			cl_int op = queue.flush();
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("flush error: ") + std::to_string(op));
			}
		}

		void sync()
		{
			cl_int op = queue.finish();
			if (op != CL_SUCCESS)
			{
				throw std::invalid_argument(std::string("finish error: ") + std::to_string(op));
			}
		}
	};
}