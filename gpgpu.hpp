// if you need all OpenCL 2.0 devices selected, then add following line before including this file  (300 for OpenCL 3.0)
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <iostream>
#include "platform.h"
#include "device.h"
#include "context.h"
#include "worker.h"
#include "kernel.h"
#include "command-queue.h"
#include "computer.h"
// todo: add error-checking for all operations