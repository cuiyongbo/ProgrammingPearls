#include "leetcode.h"

using namespace std;

/* leetcode: 96 */
class Solution {
public:
    int numTrees(int n);
};


int Solution::numTrees(int n) {
/*
    Given n, how many structurally unique BST’s (binary search trees) that store values 1 … n?
*/
    // dp[i] means number of structurally unique BSTs that store values 1 ... i.
    // dp[i] = sum(dp[j]*dp[i-j-1]), 0<=j<i // left, root, right
    vector<int> dp(n+1, 0);
    dp[0] = 1; // trivial case
    for (int i=1; i<=n; ++i) {
        for (int j=0; j<i; ++j) {
            dp[i] += dp[j] * dp[i-j-1]; // [left(0,j-1), root(j), right(j+1, i)]
        }
    }
    return dp[n];
}


void numTrees_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.numTrees(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running numTrees tests:");
    TIMER_START(numTrees);
    numTrees_scaffold(0, 1);
    numTrees_scaffold(1, 1);
    numTrees_scaffold(2, 2);
    numTrees_scaffold(3, 5);
    numTrees_scaffold(4, 14);
    TIMER_STOP(numTrees);
    SPDLOG_WARN("numTrees tests use {} ms", TIMER_MSEC(numTrees));
}
