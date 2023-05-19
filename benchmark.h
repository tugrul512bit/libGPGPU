#pragma once
#pragma once
//RAII style benchmark utility
#include<chrono>
class Bench
{
public:
	Bench(size_t* targetPtr)
	{
		target = targetPtr;
		t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		t2 = std::chrono::nanoseconds(0);
	}

	~Bench()
	{
		t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		*target = t2.count() - t1.count();
	}
private:
	size_t* target;
	std::chrono::nanoseconds t1, t2;
};