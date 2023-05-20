#include "kernel.h"
namespace GPGPU_LIB
{
	Kernel::Kernel(Context con, std::string kernelCode, std::string kernelName )
	{
		isRunning = false;
		code = kernelCode;
		name = kernelName;
		context = con;
		if (kernelCode == "" && kernelName == "")
		{

		}
		else
		{
			cl::Program::Sources source;
			source.push_back(code);
			cl::Program program(con.context, source);
			cl_int op = 0;
			if (con.device.ver >= 300)
			{
				op = program.build(con.device.device, "-cl-std=CL3.0 -cl-mad-enable");
			}
			else if (con.device.ver >= 200)
			{
				op = program.build(con.device.device, "-cl-std=CL2.0 -cl-mad-enable");
			}
			else if (con.device.ver >= 120)
			{
				op = program.build(con.device.device, "-cl-std=CL1.2 -cl-mad-enable");
			}


			if (op == CL_SUCCESS)
			{
				kernel = cl::Kernel(program, name.c_str());
			}
			else
			{
				throw std::invalid_argument(std::string("program build error: error-code=") + getErrorString(op) + std::string(" --> ") + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(con.device.device));
			}
		}
	}
}