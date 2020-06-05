#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 72, 10 */

class Solution 
{
public:
    int minDistance(string word1, string word2);
    bool isMatch(string s, string p);

private:
    int minDistance_dp(string word1, string word2);
    int minDistance_memoization(string word1, string word2);
};

int Solution::minDistance(string word1, string word2)
{
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

int Solution::minDistance_dp(string word1, string word2)
{
    // dp[i][j] means minDistance(word1[0,i), word2[0, j))
    // dp[i][j] = dp[i-1][j-1] if word1[i-1]=word2[j-1]
    // dp[i][j] = min{dp[i][j-1], dp[i-1][j], dp[i-1][j-1]} + 1

    int m = (int)word1.size();
    int n = (int)word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for(int i=0; i<=n; i++) dp[0][i] = i; // insertion
    for(int i=0; i<=m; i++) dp[i][0] = i; // deletion

    for(int i=1; i<=m; ++i)
    {
        for(int j=1; j<=n; ++j)
        {
            if(word1[i-1] == word2[j-1])
            {
                dp[i][j] = dp[i-1][j-1];
            }
            else
            {
                dp[i][j] = std::min(dp[i-1][j-1], std::min(dp[i][j-1], dp[i-1][j]))+1;
            }
        }
    }

    return dp[m][n];
}

int Solution::minDistance_memoization(string word1, string word2)
{
    int m = (int)word1.size();
    int n = (int)word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, -1));
    function<int(int, int)> dfs = [&](int l1, int l2)
    {
        if(l1 == 0) return l2;
        if(l2 == 0) return l1;
        if(dp[l1][l2] >= 0) return dp[l1][l2];

        if(word1[l1-1] == word2[l2-1])
            dp[l1][l2] = dfs(l1-1, l2-1);
        else
            dp[l1][l2] = 1 + std::min(dfs(l1-1, l2-1),
                std::min(dfs(l1-1, l2), dfs(l1, l2-1)));
        return dp[l1][l2];
    };

    return dfs(m, n);
}

bool Solution::isMatch(string s, string p)
{
    /*
        Given an input string (s) and a pattern (p), implement regular expression matching 
        with support for '.' and '*'.

        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.
        The matching should cover the entire input string (not partial).

        Note:

            s could be empty and contains only lowercase letters a-z.
            p could be empty and contains only lowercase letters a-z, and characters like . or  *.

        Example 2:
            Input:
            s = "aa"
            p = "a*"
            Output: true
            Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

        Example 3:
            Input:
            s = "ab"
            p = ".*"
            Output: true
            Explanation: ".*" means "zero or more (*) of any character (.)".
    */

    int n1 = (int)s.size();
    int n2 = (int)p.size();

    function<bool(int, int)> dfs = [&](int l1, int l2)
    {
        if(l2 == n2) return l1 == n1;
        if(l2+1 != n2 && p[l2+1] == '*') {
            int i= (l1 > 0) ? (l1-1): 0;
            while(i<n1 && (s[i] == p[l2] || p[l2] == '.'))
            {
                if(dfs(i+1, l2+2)) return true;
                ++i;
            }
            return false;
        } else {
            // s comes to end
            if(l1 == n1) return false;

            // . or matched position
            if(s[l1] == p[l2] || p[l2] == '.')
                return dfs(l1+1, l2+1);
        }
        return false;
    };

    return dfs(0, 0);
}

void minDistance_scaffold(string input1, string input2, int expectedResult)
{
    Solution ss;
    int actual = ss.minDistance(input1, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void isMatch_scaffold(string input1, string input2, bool expectedResult)
{
    Solution ss;
    int actual = ss.isMatch(input1, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running minDistance tests:";
    TIMER_START(minDistance);
    minDistance_scaffold("hello", "hope", 4);
    TIMER_STOP(minDistance);
    util::Log(logESSENTIAL) << "minDistance using " << TIMER_MSEC(minDistance) << " milliseconds";

    util::Log(logESSENTIAL) << "Running isMatch tests:";
    TIMER_START(isMatch);
    isMatch_scaffold("aa", "a", false);
    isMatch_scaffold("aa", "a*", true);
    isMatch_scaffold("ab", ".*", true);
    isMatch_scaffold("aab", "c*a*b*", true);
    isMatch_scaffold("mississippi", "mis*is*p*.", false);
    TIMER_STOP(isMatch);
    util::Log(logESSENTIAL) << "isMatch using " << TIMER_MSEC(isMatch) << " milliseconds";
}
