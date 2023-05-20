#include "benchmark.h"
namespace GPGPU
{

		Bench::Bench(size_t* targetPtr)
		{
			target = targetPtr;
			t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			t2 = std::chrono::nanoseconds(0);
		}

		Bench::~Bench()
		{
			t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			*target = t2.count() - t1.count();
		}
}