#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <functional>
#include <vector>

int main()
{
    std::vector<int> v(10, 2);
    std::partial_sum(v.begin(), v.end(), v.begin());
    std::cout << "Among the numbers: ";
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";

    if(std::all_of(v.begin(), v.end(), [](int i) {return i%2==0;}))
    {
        std::cout << "all numbers are even\n";
    }

    if(std::none_of(v.begin(), v.end(), 
        std::bind(std::modulus<int>(), std::placeholders::_1, 2)))
    {
        std::cout << "None of them are odd\n";
    }

    struct DivisibleBy
    {
        const int d;
        DivisibleBy(int n): d(n) {}
        bool operator()(int n) const {return n%d==0;}
    };

    if(std::any_of(v.begin(), v.end(), DivisibleBy(7)))
    {
        std::cout << "At least one number is divisible by 7\n";
    }

    return 0;
}

