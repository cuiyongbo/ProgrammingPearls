#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 131, 89 */

class Solution 
{
public:
    vector<vector<string>> partition(string s);
    vector<int> grayCode(int n);

private:
    vector<vector<string>> partition_backtrace(string s);
};

vector<vector<string>> Solution::partition(string s)
{
    /*
        Given a string s, partition s such that every substring of the partition is a palindrome.
        Return all possible palindrome partitioning of s.

        Input: "aab"
        Output:
        [
          ["aa","b"],
          ["a","a","b"]
        ]
    */

    return partition_backtrace(s);
}

vector<vector<string>> Solution::partition_backtrace(string s)
{
    auto isPalindrome = [&](int l, int r)
    {
        while(l < r && s[l] == s[r])
        {
            l++; r--;
        }
        return s[l] == s[r];
    };

    int len = (int)s.size();

    vector<string> cur;
    vector<vector<string>> ans;
    function<void(int)> backtrace = [&](int start)
    {
        if(start == len)
        {
            ans.push_back(cur);
            return;
        }

        for(int i=start; i<len; i++)
        {
            if(!isPalindrome(start, i)) continue;
            cur.push_back(s.substr(start, i-start+1));
            backtrace(i+1);
            cur.pop_back();
        }
    };

    backtrace(0);
    return ans;
}

vector<int> Solution::grayCode(int n)
{
    /*
        The gray code is a binary numeral system where two successive values differ in only one bit.

        Given a non-negative integer n representing the total number of bits in the code, 
        print the sequence of gray code. A gray code sequence must begin with 0.   
    */

    // dp[i] = dp[i-1] + {x|(1<<(i-1)) for x in reversed(dp[i-1])}
    // dp[0] = {0}

    vector<vector<int>> dp(n+1);
    dp[0] = {0};
    for(int i=1; i<=n; ++i)
    {
        dp[i] = dp[i-1];
        for(int j=dp[i-1].size()-1; j >= 0; --j)
        {
            dp[i].push_back(dp[i-1][j] | (1<<(i-1)));
        }
    }
    return dp[n];
}

void partition_scaffold(string input, string expectedResult)
{
    Solution ss;
    auto expected = stringTo2DArray<string>(expectedResult);
    auto actual = ss.partition(input);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        for(const auto& s1: actual)
        {
            for(const auto& s2: s1)
                util::Log(logERROR) << "Actual: " << s2;
        }
    }
}

void grayCode_scaffold(int input, string expectedResult)
{
    Solution ss;
    auto expected = stringTo1DArray<int>(expectedResult);
    auto actual = ss.grayCode(input);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << numberVectorToString<int>(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running partition tests:";
    TIMER_START(partition);
    partition_scaffold("aab", "[[a,a,b],[aa,b]]");
    TIMER_STOP(partition);
    util::Log(logESSENTIAL) << "partition using " << TIMER_MSEC(partition) << " milliseconds";

    util::Log(logESSENTIAL) << "Running grayCode tests:";
    TIMER_START(grayCode);
    grayCode_scaffold(2, "[0,1,3,2]");
    grayCode_scaffold(0, "[0]");
    TIMER_STOP(grayCode);
    util::Log(logESSENTIAL) << "grayCode using " << TIMER_MSEC(grayCode) << " milliseconds";
}
