#include "leetcode.h"

using namespace std;

int maxSubArray(vector<int>& input)
{
    long ans = INT32_MIN, curSum = 0;
    for(int i=0; i<input.size(); ++i)
    {
        curSum = max(curSum+input[i], (long)input[i]);
        ans = max(ans, curSum);
    }
    return ans;
}

void printVector(vector<int>& input)
{
    copy(input.begin(), input.end(), ostream_iterator<int>(cout, " "));
    cout << "\n";
}

int main()
{
    vector<int> input {1, -2, 3, 10, -4, 7, 2, -5};
    printVector(input);
    cout << maxSubArray(input) << "\n";
    return 0;
}
