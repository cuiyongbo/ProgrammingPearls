#include <iostream>
#include <thread>

thread_local int tt = 0;

void threadFunc(int i) {
	tt = i;
	tt++;
	std::cout << "thread: " << std::this_thread::get_id() << ", tt: " <<  tt << "\n";
}

void naive_test() {
	tt = 9;
	std::thread t1(threadFunc, 1);	
	std::thread t2(threadFunc, 3);	
	t1.join();
	t2.join();
	std::cout << "main_thread: " << std::this_thread::get_id() << ", tt: " <<  tt << "\n";
}

class TeaCup {
public:
	TeaCup() {
		m_counter = rand();
		std::cout << "thread: " << std::this_thread::get_id() << ", TeaCup(), m_counter=" << m_counter << "\n";

	}
	~TeaCup() {
		std::cout << "thread: " << std::this_thread::get_id() << ", ~TeaCup()" << "\n";
	}
	void nothing() {
		m_counter++;
		std::cout << "thread: " << std::this_thread::get_id() << ", nothing(), m_counter=" << m_counter << "\n";
	}
private:
	int m_counter;
};

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

int main() {
	//naive_test();
	tea_test();

