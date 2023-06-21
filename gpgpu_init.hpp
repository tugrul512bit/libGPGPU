#ifndef GPGPU_INIT_LIB
#define GPGPU_INIT_LIB

#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif
#define CL_HPP_TARGET_OPENCL_VERSION (120>CL_HPP_MINIMUM_OPENCL_VERSION?120:CL_HPP_MINIMUM_OPENCL_VERSION)
#include <CL/opencl.hpp>
#include <string>

const char* getErrorString0(cl_int error);

std::string getErrorString(cl_int errorCode);



#include "benchmark.h"
#include <exception>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <vector>

#endif