#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 10, 72 */

class Solution {
public:
    bool isMatch(string s, string p);
    int minDistance(string word1, string word2);
private:
    int minDistance_dp(string word1, string word2);
    int minDistance_memoization(string word1, string word2);
};

int Solution::minDistance(string word1, string word2) {
/*
    Given two words word1 and word2, find the minimum number of steps required 
    to convert word1 to word2. (each operation is counted as 1 step.)
    You have the following 3 operations permitted on a word:
        a) Insert a character
        b) Delete a character
        c) Replace a character
*/
    // return minDistance_dp(word1, word2);
    return minDistance_memoization(word1, word2);
}

int Solution::minDistance_dp(string word1, string word2) {
    // dp[i][j] means minDistance(word1[0,i), word2[0, j)). i, j are not inclusive
    // dp[i][j] = dp[i-1][j-1] if word1[i-1]=word2[j-1] else 
    //      dp[i][j] = min{dp[i][j-1], dp[i-1][j], dp[i-1][j-1]} + 1
    int m = word1.size();
    int n = word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, INT32_MAX));
    // trivial cases:
    for (int i=0; i<=n; i++) {
        dp[0][i] = i; // insertion
    }
    for (int i=0; i<=m; i++) {
        dp[i][0] = i; // deletion
    }
    // recursion: 
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1]; // no operation
            } else {
                dp[i][j] = min(dp[i][j], dp[i-1][j-1]+1); // replacement
                dp[i][j] = min(dp[i][j], dp[i][j-1]+1); // insertion
                dp[i][j] = min(dp[i][j], dp[i-1][j]+1); // deletion
            }
        }
    }
    return dp[m][n];
}

int Solution::minDistance_memoization(string word1, string word2) {
    int m = word1.size();
    int n = word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, -1));
    function<int(int, int)> dfs = [&] (int l1, int l2) {
        if (l1 == 0) {
            return l2;
        }
        if (l2 == 0) {
            return l1;
        }
        if (dp[l1][l2] >= 0) {
            return dp[l1][l2];
        }
        if (word1[l1-1] == word2[l2-1]) {
            dp[l1][l2] = dfs(l1-1, l2-1);
        } else {
            dp[l1][l2] = 1 + min(dfs(l1-1, l2-1), min(dfs(l1-1, l2), dfs(l1, l2-1)));
        }
        return dp[l1][l2];
    };
    return dfs(m, n);
}


bool Solution::isMatch(string s, string p) {
/*
    Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
    '.' Matches any single character. and '*' Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).
    Note:
        s could be empty or contains only lowercase letters a-z.
        p could be empty or contains only lowercase letters a-z, and characters like . or  *.
    For example, given an input: s = "ab", p = ".*", output: true, explanation: ".*" means "zero or more (*) of any character (.)".
*/

if (1) {
    int m = s.size(), n = p.size();
    // dp[i][j] means whether p[:j] matches s[:i] or not (**the right ends are inclusive**)
    vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
    dp[0][0] = true; // trivial case
    for (int j=2; j<=n; ++j) {
        if (p[j-1] == '*') { // trivial cases to match an empty string
            dp[0][j] = dp[0][j-2];
        }
    }
    // i, j are not inclusive
    for (int i=1; i<=m; ++i) {
        for (int j=1; j<=n; ++j) {
            if (p[j-1] == '.') { // a "." match
                dp[i][j] = dp[i-1][j-1];
            } else if (p[j-1] == '*') {
                // 1. Use '*' to match zero characters in s
                dp[i][j] = dp[i][j-2];
                // 2. Use '*' to match one or more characters in s
                if (s[i-1] == p[j-2] || p[j-2]=='.') {
                    dp[i][j] = dp[i][j] || dp[i-1][j];
                }
            } else { // case needing an exact match
                dp[i][j] = (s[i-1] == p[j-1]) ? dp[i-1][j-1] : false;
            }
        }
    }
    return dp[m][n];
}

{ // failure solution
    int n1 = s.size();
    int n2 = p.size();
    function<bool(int, int)> dfs = [&](int l1, int l2) {
        if (l2 == n2) {
            return l1 == n1;
        }
        if (l2+1 != n2 && p[l2+1] == '*') { // next character in pattern is a wildcard
            //int i = (l1 > 0) ? (l1-1): 0;
            int i = l1;
            while (i<n1 && (s[i] == p[l2] || p[l2] == '.')) {
                if (dfs(i+1, l2+2)) { // the wildcard at p[l2+1] match zero or more character(s)
                    return true;
                }
                ++i;
            }
        } else { // next character in pattern is either a dot or a lowercase letter
            // s comes to end
            if (l1 == n1) {
                return false;
            }
            // dot or exact match
            if (s[l1] == p[l2] || p[l2] == '.') {
                return dfs(l1+1, l2+1);
            }
        }
        return false;
    };
    return dfs(0, 0);
}

}


void minDistance_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    int actual = ss.minDistance(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


void isMatch_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    bool actual = ss.isMatch(input1, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running minDistance tests:");
    TIMER_START(minDistance);
    minDistance_scaffold("hello", "hope", 4);
    TIMER_STOP(minDistance);
    SPDLOG_WARN("minDistance tests use {} ms", TIMER_MSEC(minDistance));

    SPDLOG_WARN("Running isMatch tests:");
    TIMER_START(isMatch);
    isMatch_scaffold("aa", "a", false);
    isMatch_scaffold("aa", "a*", true);
    isMatch_scaffold("ab", ".*", true);
    isMatch_scaffold("aab", "c*a*b*", true);
    isMatch_scaffold("mississippi", "mis*is*p*.", false);
    TIMER_STOP(isMatch);
    SPDLOG_WARN("isMatch tests use {} ms", TIMER_MSEC(isMatch));
}
