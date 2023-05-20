#include "context.h"

GPGPU::Context::Context(GPGPU::Device dev)
{
	context = cl::Context(dev.device);
	device = dev;
}