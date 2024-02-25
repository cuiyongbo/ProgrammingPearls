#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 279 */

class Solution {
public:
    int numSquares(int n);
};

int Solution::numSquares(int n) {
/*
    Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
    A perfect square is an integer that is the square of an integer. e.g. 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.
*/

{
    // dp[i] means the least number of perfect square numbers which sum to i
    // dp[i] = min{dp[i-j*j]+1}, 0<j*j<=i
    vector<int> dp(n+1, INT32_MAX); dp[0] = 0;
    for (int i=1; i<=n; ++i) {
        for (int j=1; j*j<=i; ++j) {
            dp[i] = min(dp[i], dp[i-j*j]+1);
        }
    }
    return dp[n];
}

}

void numSquares_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.numSquares(input);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running numSquares tests:";
    TIMER_START(numSquares);
    numSquares_scaffold(12, 3);
    numSquares_scaffold(13, 2);
    numSquares_scaffold(9, 1);
    numSquares_scaffold(4, 1);
    TIMER_STOP(numSquares);
    util::Log(logESSENTIAL) << "numSquares using " << TIMER_MSEC(numSquares) << " milliseconds";
}
