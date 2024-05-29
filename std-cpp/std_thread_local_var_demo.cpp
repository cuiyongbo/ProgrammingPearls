#include <iostream>
#include <thread>
#include <stdint.h>

thread_local int tt = 0;
std::mutex cout_mutex;

void thread_func(int i) {
	tt = i;
	tt++;
	std::lock_guard<std::mutex> guard(cout_mutex);
	std::cout << "thread: " << std::this_thread::get_id() << ", &tt: " <<  &tt << ", tt: " <<  tt << std::endl;
}

void naive_test() {
	tt = 9;
	std::thread t1(thread_func, 1);	
	std::thread t2(thread_func, 3);	
	t1.join();
	t2.join();
	{
		std::lock_guard<std::mutex> guard(cout_mutex);
		std::cout << "main_thread: " << std::this_thread::get_id() << ", &tt: " <<  &tt << ", tt: " <<  tt << std::endl;
	}

}

class TeaCup {
public:
	TeaCup() {
		m_counter = rand();
		std::cout << "thread: " << std::this_thread::get_id() << ", TeaCup(), m_counter=" << m_counter << std::endl;

	}
	~TeaCup() {
		std::cout << "thread: " << std::this_thread::get_id() << ", ~TeaCup()" << std::endl;
	}
	void nothing() {
		m_counter++;
		std::cout << "thread: " << std::this_thread::get_id() << ", nothing(), m_counter=" << m_counter << std::endl;
	}
private:
	int m_counter;
};

// for a thread_local variable, it will be constructed as the thread begins and destructed as the thread ends
// and each thread has its own instance of the object
thread_local TeaCup tea_cup;
void cook_tea() {
	tea_cup.nothing();
}

void tea_test() {
	tea_cup.nothing();
	std::thread t1(cook_tea);	
	std::thread t2(cook_tea);	
	t1.join();
	t2.join();
}

class Jack {
private:
	// thread_local int m_days; // thread-local storage class is not valid here
	// Note that if you want to declare a class member as thread_local, it must be a static member. since it would be shared
	// by different class instances in the same thread according to its specification.
	static thread_local int m_days;
};

int main() {
	naive_test();
	//tea_test();

}
