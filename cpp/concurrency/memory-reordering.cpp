#include <atomic>
#include <future>
#include <iostream>

std::atomic<int> x;
std::atomic<int> y;

int f1()
{
	x.store(1, std::memory_order_relaxed);
	return y.load(std::memory_order_relaxed);
}

int f2()
{
	y.store(1, std::memory_order_relaxed);
	return x.load(std::memory_order_relaxed);
}

int main(int, char**)
{
	for(int i = 0; i < 100000; i++)
	{
		x = 0;
		y = 0;

		auto t1 = std::async(std::launch::async, f1);
		auto t2 = std::async(std::launch::async, f2);

		auto r1 = t1.get();
		auto r2 = t2.get();

		if(r1 == 0 && r2 == 0)
			std::cout << "T1: " << r1 << " - T2: " << r2 << '\n';
	}
}