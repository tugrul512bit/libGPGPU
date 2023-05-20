#include "context.h"

GPGPU_LIB::Context::Context(GPGPU_LIB::Device dev)
{
	context = cl::Context(dev.device);
	device = dev;
}