#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
 
int main()
{
	std::vector<int> data = { 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6 };
	auto lower = std::lower_bound(data.begin(), data.end(), 4);
	auto upper = std::upper_bound(data.begin(), data.end(), 4);
	std::cout << "Find lower bound at index: " << std::distance(data.begin(), lower) << ", value: " << *lower << '\n';
	std::cout << "Find upper bound at index: " << std::distance(data.begin(), upper) << ", value: " << *upper << '\n';
	std::cout << std::distance(lower, upper) << '\n';
	std::copy(lower, upper, std::ostream_iterator<int>(std::cout, " "));
	std::cout << '\n';
}
