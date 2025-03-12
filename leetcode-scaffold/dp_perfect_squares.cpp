#include "leetcode.h"

using namespace std;

/* leetcode: 279 */
class Solution {
public:
    int numSquares(int n);
    vector<int> findPerfectSquareSequence(int n);
};


int Solution::numSquares(int n) {
/*
    Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
    A perfect square is an integer that is the square of an integer. e.g. 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.
*/
    // dp[i] means the least number of perfect square numbers which sum to i
    // dp[i] = min{dp[i-j*j]} 1<=j*j<=i
    vector<int> dp(n+1, INT32_MAX);
    dp[0] = 0; dp[1] = 1; // trivial cases
    for (int i=1; i<=n; i++) {
        // nice catcha
        for (int j=1; j*j<=i; j++) {
            dp[i] = min(dp[i], dp[i-j*j]+1);
        }
    }
    return dp[n];
}


// how to reconstruct the sequence of perfect square numbers?
vector<int> Solution::findPerfectSquareSequence(int n) {
    vector<vector<int>> dp(n+1);
    for (int i=1; i<=n; i++) {
        for (int j=1; j*j<=i; j++) {
            if (dp[i].empty()) {
                dp[i] = dp[i-j*j];
                dp[i].push_back(j*j);
            } else if (dp[i].size() > dp[i-j*j].size()+1) {
                dp[i] = dp[i-j*j];
                dp[i].push_back(j*j);
            } else if (dp[i].size() == dp[i-j*j].size()+1) {
                // there maybe more than one valid sequences
            }
        }
    }
    SPDLOG_INFO("findPerfectSquareSequence({}): {}", n, numberVectorToString(dp[n]));
    return dp[n];
}


void numSquares_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.numSquares(input);
    ss.findPerfectSquareSequence(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running numSquares tests:");
    TIMER_START(numSquares);
    numSquares_scaffold(12, 3);
    numSquares_scaffold(13, 2);
    numSquares_scaffold(9, 1);
    numSquares_scaffold(4, 1);
    numSquares_scaffold(2, 2);
    numSquares_scaffold(3, 3);
    numSquares_scaffold(5, 2);
    TIMER_STOP(numSquares);
    SPDLOG_WARN("numSquares tests use {} ms", TIMER_MSEC(numSquares));
}
