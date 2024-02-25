#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>

using namespace std;

template<class T>
void printVector(vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " "));
    cout << "\n";
} 

void accumulate_test()
{
    vector<int> vi {1,2,3,4,5,6};
    printVector(vi);
    cout << "Accumulate sum: " << accumulate(vi.begin(), vi.end(), 0) << "\n";
    cout << "Accumulate product: " << accumulate(vi.begin(), vi.end(), 1, multiplies<int>()) << "\n";
}

void partial_sum_demo()
{
    vector<int> vi {2,2,2,2,2,2,2,2};
    printVector(vi);
    cout << "Prefix sum: ";
    partial_sum(vi.begin(), vi.end(), ostream_iterator<int>(cout, " "));
    cout << "\n";

    cout << "The first " << vi.size() << " powers of 2: \n";
    partial_sum(vi.begin(), vi.end(), vi.begin(), multiplies<int>());
    printVector(vi);
}

int main()
{
    accumulate_test();
    partial_sum_demo();

    return 0;
}
