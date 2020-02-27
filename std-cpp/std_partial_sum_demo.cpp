#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>

using namespace std;

int main()
{
    vector<int> vi(10);
    std::iota(vi.begin(), vi.end(), 1);

    auto printVector = [](vector<int>& vv)
    {
        std::copy(vv.begin(), vv.end(), std::ostream_iterator<int>(cout, " "));
        cout << "\n";
    };

    printVector(vi);

    vector<int> svi(vi.size());
    std::partial_sum(vi.begin(), vi.end(), svi.begin());
    printVector(svi);

    return 0;
}
