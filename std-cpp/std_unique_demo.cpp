#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cctype>
#include <iterator>
 
int main() 
{
    // remove duplicate elements
    std::vector<int> v{1,2,3,1,2,3,3,4,5,4,5,6,7};
    std::sort(v.begin(), v.end()); // 1 1 2 2 3 3 3 4 4 5 5 6 7 
    auto last = std::unique(v.begin(), v.end());
	std::cout << "size: " << v.size() << '\n';
    // v now holds {1 2 3 4 5 6 7 x x x x x x}, where 'x' is indeterminate
    v.erase(last, v.end()); 
	std::cout << "size: " << v.size() << '\n';
	std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
   // for (int i : v)
   //   std::cout << i << " ";
    std::cout << "\n";
}

