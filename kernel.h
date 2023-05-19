#pragma once
#include <string>
#include <iostream>
#include <map>
#include "gpgpu_init.hpp"
#include "context.h"
#include "device.h"
#include "parameter.h"


namespace GPGPU
{

	struct Kernel
	{

		cl::Kernel kernel;
		std::string name;
		std::string code;
		Context context;
		bool isRunning;
		std::map<std::string, Parameter> mapParameterNameToParameter;
		Kernel(Context con = Context(), std::string kernelCode = "", std::string kernelName = "")
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
				cl_int op = program.build(con.device.device, "-cl-std=CL1.2 -cl-mad-enable");
				if (op == CL_SUCCESS)
				{
					kernel = cl::Kernel(program, name.c_str());
				}
				else
				{
					throw std::invalid_argument(std::string("program build error: error-code=") + std::to_string(op) + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(con.device.device));
				}
			}
		}



	};

}