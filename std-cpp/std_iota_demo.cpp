#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>


using namespace std;

int main()
{
	    vector<int> v(14, 0);
		
		std::iota(v.begin(), v.end(), 0);		
        std::copy(v.begin(), v.end(), std::ostream_iterator<int>(cout, " "));
        cout << "\n";
}

