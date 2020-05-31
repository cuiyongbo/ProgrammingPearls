#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 300, 673, 1048, 674, 128 */

class Solution 
{
public:
    int lengthOfLIS(vector<int>& nums);
    int findNumberOfLIS(vector<int>& nums);
    int findLengthOfLCIS(vector<int>& nums);
    int longestStrChain(vector<string>& words);
    int longestConsecutive(vector<int> &num);
};

int Solution::lengthOfLIS(vector<int>& nums)
{
    /*
        Given an unsorted array of integers, find the length of longest increasing subsequence.

        For example,
        Given [10, 9, 2, 5, 3, 7, 101, 18],
        The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. 
        Note that there may be more than one LIS combination, it is only necessary for you to return the length.

        Your algorithm should run in O(n2) complexity.
    */

    // dp[i] means lengthOfLIS ending with nums[i], nums[i] must be used
    // dp[i] = max{dp[j] + nums[j] < nums[i]}, 0 <= j <i
    int ans = INT32_MIN;
    int n = (int)nums.size();
    vector<int> dp(n, 1);
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<i; ++j)
        {
            if(nums[j] < nums[i]) 
                dp[i] = std::max(dp[i], dp[j]+1);
        }
        ans = std::max(dp[i], ans);
    }
    return ans;
}

int Solution::findNumberOfLIS(vector<int>& nums)
{
    /*
        Given an unsorted array of integers, 
        find the number of longest increasing subsequence.
    */

    // dp[i] means lengthOfLIS ending with nums[i], nums[i] must be used
    // dp[i] = max{dp[j] + nums[j] < nums[i]}, 0 <= j <i
    int maxLen = INT32_MIN;
    int n = (int)nums.size();
    vector<int> dp(n, 1);
    vector<int> count(n, 1);
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<i; ++j)
        {
            if(nums[j] < nums[i]) 
            {
                if(dp[j]+1 > dp[i])
                {
                    dp[i] = dp[j]+1;
                    count[i] = count[j];
                }
                else if(dp[j]+1 == dp[i])
                {
                    count[i] += count[j];
                }
            }
        }
        maxLen = std::max(dp[i], maxLen);
    }

    int ans = 0;
    for(int i=0; i<n; i++)
    {
        if(dp[i] == maxLen) ans += count[i];
    }
    return ans;
}

int Solution::findLengthOfLCIS(vector<int>& nums)
{
    /*
        Given an unsorted array of integers, 
        find the length of longest continuous increasing subsequence.

        Example 1:
        Input: [1,3,5,4,7]
        Output: 3
        Explanation: The longest continuous increasing subsequence is [1,3,5], its length is 3. 
        Even though [1,3,5,7] is also an increasing subsequence, 
        it’s not a continuous one where 5 and 7 are separated by 4.
    */

    // dp[i] means lengthOfLCIS ending with nums[i], nums[i] must be used
    // dp[i] = 1 if nums[i] <= nums[j]
    // dp[i] = dp[j] + 1 if nums[i] > nums[j]

    if(nums.empty()) return 0;

    int maxLen = 1;
    int n = (int)nums.size();
    vector<int> dp(n, 1);
    for(int i=1; i<n; ++i)
    {
        if(nums[i] > nums[i-1])
        {
            dp[i] = dp[i-1] + 1;
        }
        maxLen = std::max(dp[i], maxLen);
    }
    return maxLen;
}

int Solution::longestStrChain(vector<string>& words)
{
    /*
        Given a list of words, each word consists of English lowercase letters.

        Let's say word1 is a predecessor of word2 if and only if we can add exactly 
        one letter anywhere in word1 to make it equal to word2.  
        For example, "abc" is a predecessor of "abac".

        A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, 
        where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on.

        Return the longest possible length of a word chain with words chosen from the given list of words.

        Example 1:
            Input: ["a","b","ba","bca","bda","bdca"]
            Output: 4
            Explanation: one of the longest word chain is "a","ba","bda","bdca".
    */

    auto isPredecessor = [&](const string& pre, const string& next)
    {
        int s1 = pre.size();
        int s2 = next.size();
        if(s1 + 1 != s2) return false;

        int i=0, j=0;
        while(i<s1 && j<s2)
        {
            if(pre[i] == next[j])
            {
                i++; j++;
            }
            else
            {
                j++;
            }
        }
        return i==s1 && j==s2;
    };

    int ans = 0;
    int n = words.size();
    vector<int> dp(n, 1);
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<i; j++)
        {
            if(isPredecessor(words[j], words[i]))
            {
                dp[i] = std::max(dp[i], dp[j]+1);
            }
            ans = std::max(dp[i], ans);
        }
    }
    return ans;
}

int Solution::longestConsecutive(vector<int>& nums)
{
    /*
        Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

        For example,
        Given [100, 4, 200, 1, 3, 2],
        The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
        Your algorithm should run in O(n) complexity.
    */

    // brute force: first sort the arrary, then count longestConsecutive

    int ans = 0;
    unordered_map<int, int> m;
    for(const auto& n: nums)
    {
        if(m.count(n)) continue;
        if(m.count(n+1) && m.count(n-1))
        {
            int l = m[n-1];
            int r = m[n+1];
            m[n] = m[n-l] = m[n+r] = l+r+1;
        }
        else if(m.count(n-1))
        {
            int l = m[n-1];
            m[n] = m[n-l] = l+1;
        }
        else if(m.count(n+1))
        {
            int r = m[n+1];
            m[n] = m[n+r] = r+1;
        }
        else
        {
            m[n] = 1;
        }
        ans = std::max(ans, m[n]);
    }
    return ans;
}

void lengthOfLIS_scaffold(string input, int expectedResult)
{
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.lengthOfLIS(nums);
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

void findNumberOfLIS_scaffold(string input, int expectedResult)
{
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.findNumberOfLIS(nums);
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

void findLengthOfLCIS_scaffold(string input, int expectedResult)
{
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.findLengthOfLCIS(nums);
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

void longestConsecutive_scaffold(string input, int expectedResult)
{
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.longestConsecutive(nums);
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

void longestStrChain_scaffold(string input, int expectedResult)
{
    Solution ss;
    auto words = stringTo1DArray<string>(input);
    int actual = ss.longestStrChain(words);
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

    util::Log(logESSENTIAL) << "Running lengthOfLIS tests:";
    TIMER_START(lengthOfLIS);
    lengthOfLIS_scaffold("[10, 9, 2, 5, 3, 7, 101, 18]", 4);
    lengthOfLIS_scaffold("[10, 9, 2, 5, 3, 4, 7, 101, 18]", 5);
    TIMER_STOP(lengthOfLIS);
    util::Log(logESSENTIAL) << "lengthOfLIS using " << TIMER_MSEC(lengthOfLIS) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findNumberOfLIS tests:";
    TIMER_START(findNumberOfLIS);
    findNumberOfLIS_scaffold("[10, 9, 2, 5, 3, 7, 101, 18]", 4);
    findNumberOfLIS_scaffold("[2,2,2,2,2]", 5);
    findNumberOfLIS_scaffold("[1,3,5,4,7]", 2);
    TIMER_STOP(findNumberOfLIS);
    util::Log(logESSENTIAL) << "findNumberOfLIS using " << TIMER_MSEC(findNumberOfLIS) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findLengthOfLCIS tests:";
    TIMER_START(findLengthOfLCIS);
    findLengthOfLCIS_scaffold("[2,2,2,2,2]", 1);
    findLengthOfLCIS_scaffold("[1,3,5,4,7]", 3);
    TIMER_STOP(findLengthOfLCIS);
    util::Log(logESSENTIAL) << "findLengthOfLCIS using " << TIMER_MSEC(findLengthOfLCIS) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestConsecutive tests:";
    TIMER_START(longestConsecutive);
    longestConsecutive_scaffold("[2,2,2,2,2]", 1);
    longestConsecutive_scaffold("[1,3,5,4,7]", 3);
    longestConsecutive_scaffold("[100,4,200,1,3,2]", 4);
    TIMER_STOP(longestConsecutive);
    util::Log(logESSENTIAL) << "longestConsecutive using " << TIMER_MSEC(longestConsecutive) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestStrChain tests:";
    TIMER_START(longestStrChain);
    longestStrChain_scaffold("[a,b,ba,bca,bda,bdca]", 4);
    TIMER_STOP(longestStrChain);
    util::Log(logESSENTIAL) << "longestStrChain using " << TIMER_MSEC(longestStrChain) << " milliseconds";
}
