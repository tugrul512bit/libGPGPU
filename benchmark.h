#pragma once
#ifndef GPGPU_BENCH_LIB
#define GPGPU_BENCH_LIB


#include<chrono>
namespace GPGPU
{
	//RAII style benchmark utility

	class Bench
	{
	public:
		Bench(size_t* targetPtr);

		~Bench();
	private:
		size_t* target;
		std::chrono::nanoseconds t1, t2;
	};
}

#endif // !GPGPU_BENCH_LIB