#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

template<class T>
void printVector(std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}

template<class ForwardIt>
void quicksort(ForwardIt first, ForwardIt last)
{
	size_t count = std::distance(first, last);
	if(count < 2)
		return;

	ForwardIt p = std::next(first, count/2);
	//auto mid1 = std::partition(first, last, [p](const typename ForwardIt::value_type& elem) {return elem < *p;});
	//auto mid2 = std::partition(mid1, last, [p](const typename ForwardIt::value_type& elem) {return elem == *p;});
	auto mid1 = std::partition(first, last, std::bind2nd(std::less<typename ForwardIt::value_type>(), *p));
	auto mid2 = std::partition(mid1, last, std::bind1st(std::equal_to<typename ForwardIt::value_type>(), *p));
	
	quicksort(first, mid1);
	quicksort(mid2, last);
}

int main()
{
	std::vector<int> vi {1,3,5,2,6,4,8,7,9,4};
	std::cout << "Originally: ";
	printVector(vi);

	quicksort(vi.begin(), vi.end());

	std::cout << "quicksort: ";
	printVector(vi);

	return 0;
}

