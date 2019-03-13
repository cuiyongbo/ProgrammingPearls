#include <iostream>
#include <thread>

thread_local int tt = 0;

void threadFunc(int i)
{
	tt = i;
	tt++;
	std::cout << tt << "\n";
}

int main()
{
	tt = 9;
	std::thread t1(threadFunc, 1);	
	std::thread t2(threadFunc, 3);	

	t1.join();
	t2.join();
	
	std::cout << tt << "\n";
}
