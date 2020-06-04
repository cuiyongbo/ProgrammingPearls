#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 96 */

class Solution 
{
public:
    int numTrees(int n);
};

int Solution::numTrees(int n)
{
    /*
        Given n, how many structurally unique BST’s (binary search trees) that store values 1 … n?
    */

    // dp[i] = sum(dp[j] * dp[i-j-1]), 0<= j < i
    // root: 1 node
    // left: j node(s)
    // right: i-j-1 node(s)

    if(n <= 0) return 0;
    vector<int> dp(n+1, 0);
    dp[0] = 1;
    for(int i=0; i<=n; ++i)
    {
        for(int j=0; j < i; ++j)
        {
            dp[i] += dp[j] * dp[i-j-1];
        }
    }
    return dp[n];
}

void numTrees_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.numTrees(input);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running numTrees tests:";
    TIMER_START(numTrees);
    numTrees_scaffold(3, 5);
    TIMER_STOP(numTrees);
    util::Log(logESSENTIAL) << "numTrees using " << TIMER_MSEC(numTrees) << " milliseconds";
}
