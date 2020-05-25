#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 198, 213, 309, 740, 790, 801 */

class Solution 
{
public:
    int numSquares(int n);
};

int Solution::numSquares(int n)
{
    /*
        Given a positive integer n, find the least number 
        of perfect square numbers (for example, 1, 4, 9, 16, ...) 
        which sum to n.
    */

    // dp[i] = min{dp[i-j*j]+1}, 1<= j*j <= n
    vector<int> dp(n+1, INT32_MAX >> 1);
    dp[0] = 0;
    for(int i=0; i<=n; ++i)
    {
        for(int j=1; j*j<=i; ++j)
        {
            dp[i] = std::min(dp[i], dp[i-j*j]+1);
        }
    }
    return dp[n];
}

void numSquares_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.numSquares(input);
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

    util::Log(logESSENTIAL) << "Running numSquares tests:";
    TIMER_START(numSquares);
    numSquares_scaffold(12, 3);
    numSquares_scaffold(13, 2);
    TIMER_STOP(numSquares);
    util::Log(logESSENTIAL) << "numSquares using " << TIMER_MSEC(numSquares) << " milliseconds";
}
