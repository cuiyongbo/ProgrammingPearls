#include <vector>
#include <list>
#include <forward_list>

#include <stdio.h>
#include <time.h>

static const int itemNum = 1000000;

int main(int argc, char const *argv[])
{
	clock_t start, during;

	start = clock();
	std::vector<int> vi;
	for (int i = 0; i < itemNum; i++)
		vi.push_back(i);
	during = clock() - start;
	printf("Operation: vector::push_back, Count: %d, time using: %ld ms.\n", itemNum, during);

	start = clock();
	std::list<int> li1;
	for (int i = 0; i < itemNum; i++)
		li1.push_back(i);
	during = clock() - start;
	printf("Operation: list::push_back, Count: %d, time using: %ld ms.\n", itemNum, during);

	start = clock();
	std::list<int> li2;
	for (int i = 0; i < itemNum; i++)
		li2.push_front(i);
	during = clock() - start;
	printf("Operation: list::push_front, Count: %d, time using: %ld ms.\n", itemNum, during);

	start = clock();
	std::forward_list<int> li3;
	for (int i = 0; i < itemNum; i++)
		li3.push_front(i);
	during = clock() - start;
	printf("Operation: forward_list::push_front, Count: %d, time using: %ld ms.\n", itemNum, during);
	
	return 0;
}