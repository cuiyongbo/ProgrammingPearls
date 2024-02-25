#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>

template<class T>
void printVector(std::vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}

int main()
{
    std::vector<int> v{5, 6, 4, 3, 2, 6, 7, 9, 3};
    std::cout << "Originally\n";
	printVector(v);

    std::nth_element(v.begin(), v.end(), v.end(), std::greater<int>());
    std::cout << "nth_element(v.end())\n";
	printVector(v);

    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    std::cout << "The median is " << v[v.size()/2] << '\n';
	printVector(v);

    std::nth_element(v.begin(), v.begin()+1, v.end(), std::greater<int>());
    std::cout << "The second largest element is " << v[1] << '\n';
	printVector(v);
}
