#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 300, 673, 674, 1048, 128, 1218 */

class Solution {
public:
    int lengthOfLongestIncreasingSubsequence(vector<int>& nums);
    int findNumberOfLongestIncreasingSubsequence(vector<int>& nums);
    int findLengthOfLongestContinuousIncreasingSubsequence(vector<int>& nums);
    int longestStrChain(vector<string>& words);
    int longestConsecutive(vector<int> &num);
    int longestSubsequence(vector<int>& arr, int d);
};

int Solution::longestSubsequence(vector<int>& arr, int d) {
/*
    Given an integer array arr and an integer difference d, return the length of the longest subsequence in arr 
    which is an arithmetic sequence such that the difference between adjacent elements in the subsequence equals difference.

    Example 1:
        Input: arr = [1,2,3,4], difference = 1
        Output: 4
        Explanation: The longest arithmetic subsequence is [1,2,3,4].
    Example 2:
        Input: arr = [1,3,5,7], difference = 1
        Output: 1
        Explanation: The longest arithmetic subsequence is any single element.
*/

{
    // dp[i] means longestSubsequence ending with arr[i] 
    // dp[i] = max{dp[j]+1} for j in [0,i-1] if arr[i]-arr[j]==d else 1
    int ans = INT32_MIN;
    int n = arr.size();
    vector<int> dp(n, 1);
    for (int i=1; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            if (arr[i]-arr[j] == d) {
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}

}

int Solution::lengthOfLongestIncreasingSubsequence(vector<int>& nums) {
/*
    Given an unsorted array of integers, find the length of the longest increasing subsequence.
    For example, Given [10, 9, 2, 5, 3, 7, 101, 18], The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. 
    Note that there may be more than one LIS combination, it is only necessary for you to return the length.
*/
    // dp[i] means lengthOfLongestIncreasingSubsequence ending with nums[i]
    // dp[i] = max{dp[j]+1} if(nums[j] < nums[i]), 0 <= j <i
    int ans = INT32_MIN;
    int n = nums.size();
    vector<int> dp(n, 1);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}

int Solution::findNumberOfLongestIncreasingSubsequence(vector<int>& nums) {
/*
    Given an unsorted array of integers, find the number of the longest increasing subsequence.
*/
    int maxLen = 0;
    int n = nums.size();
    // dp[i] means lengthOfLongestIncreasingSubsequence ending with nums[i]
    // dp[i] = max{dp[j] + nums[j] < nums[i]}, 0 <= j <i
    vector<int> dp(n, 1);
    // count[i] means NumberOfLongestIncreasingSubsequence ending with nums[i]
    vector<int> count(n, 1);
    for (int i=1; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            if (nums[j] < nums[i]) { // need update intermediate vars
                if (dp[j]+1 > dp[i]) {
                    dp[i] = dp[j]+1;
                    count[i] = count[j];
                } else if (dp[j]+1 == dp[i]) {
                    count[i] += count[j];
                } else {
                    // do nothing
                }
            }
        }
        maxLen = max(maxLen, dp[i]);
    }

    int ans = 0;
    for (int i=0; i<n; ++i) {
        if (maxLen == dp[i]) {
            ans += count[i];
        }
    }
    return ans;
}

int Solution::findLengthOfLongestContinuousIncreasingSubsequence(vector<int>& nums) {
/*
    Given an unsorted array of integers, find the length of longest continuous increasing subsequence.

    Example 1:
        Input: [1,3,5,4,7]
        Output: 3
    Explanation: The longest continuous increasing subsequence is [1,3,5], its length is 3. 
    Even though [1,3,5,7] is also an increasing subsequence, it’s not a continuous one where 5 and 7 are separated by 4.
*/

    // dp[i] means lengthOfLCIS ending with nums[i]
    // dp[i] = 1 if nums[i] <= nums[i-1] else
    //  dp[i] = dp[i-1] + 1
    int ans = INT32_MIN;
    int n = nums.size();
    vector<int> dp(n, 1);
    for (int i=1; i<n; ++i) {
        if (nums[i] > nums[i-1]) {
            dp[i] = dp[i-1] + 1;
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}

int Solution::longestStrChain(vector<string>& words) {
/*
    Given a list of words, each word consists of English lowercase letters.
    Let's say word1 is a predecessor of word2 if and only if we can add exactly one letter anywhere in word1 to make it equal to word2. For example, "abc" is a predecessor of "abac".
    A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on.
    Return the longest possible length of a word chain with words chosen from the given list of words.

    Example 1:
        Input: ["a","b","ba","bca","bda","bdca"]
        Output: 4
        Explanation: one of the longest word chain is "a","ba","bda","bdca".

    Example 2:
        Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
        Output: 5
        Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].

    Hint: first sort words by length, then we can work out the longestStrChain ending with words[i]
    by dp[i] = max{dp[j]+1} if (is_predecessor(words[j], word[i])) 0<=j<i
*/
    auto is_predecessor = [] (const string& pre, const string& next) {
        int s1 = pre.size();
        int s2 = next.size();
        if (s1 + 1 != s2) {
            return false;
        }
        int i=0, j=0, diff=0;
        while (i<s1 && j<s2 && diff < 2) {
            if (pre[i] == next[j]) {
                i++; j++;
            } else {
                j++; diff++;
            }
        }
        return diff <= 1;
    };
    sort(words.begin(), words.end(), [&](const string& l, const string& r) {
            return l.size() < r.size();});
    int ans = 0;
    int n = words.size();
    vector<int> dp(n, 1);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            if (is_predecessor(words[j], words[i])) {
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}

int Solution::longestConsecutive(vector<int>& nums) {
/*
    Given an unsorted array of integers, find the length of the longest consecutive element sequence.
    For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
    Your algorithm should run in O(n) complexity.
*/
    int ans = 0;
    map<int, int> m; // element, longestConsecutive containing element
    for (auto n: nums) {
        if (m.count(n) == 1) { // IMPORTANT, make sure only end nodes will be used hereafter
            continue;
        }
        // n hasn't appeared so far
        bool left = m.count(n-1) != 0;
        bool right = m.count(n+1) != 0;
        if (left && right) {
            int l = m[n-1]; // length of subarray ending with n-1 
            int r = m[n+1]; // length of subarray beginning with n+1
            m[n] = m[n-l] = m[n+r] = l+r+1;
        } else if (left) {
            int l = m[n-1]; // length of subarray ending with n-1
            m[n] = m[n-l] = l+1;
        } else if (right) {
            int r = m[n+1]; // length of subarray beginning with n+1
            m[n] = m[n+r] = r+1;
        } else {
            m[n] = 1;
        }
        ans = max(ans, m[n]);
    }
    return ans;
}

void lengthOfLongestIncreasingSubsequence_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.lengthOfLongestIncreasingSubsequence(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findNumberOfLongestIncreasingSubsequence_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.findNumberOfLongestIncreasingSubsequence(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findLengthOfLCIS_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.findLengthOfLongestContinuousIncreasingSubsequence(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void longestConsecutive_scaffold(string input, int expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    int actual = ss.longestConsecutive(nums);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void longestStrChain_scaffold(string input, int expectedResult) {
    Solution ss;
    auto words = stringTo1DArray<string>(input);
    int actual = ss.longestStrChain(words);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void longestSubsequence_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.longestSubsequence(nums, input2);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running lengthOfLongestIncreasingSubsequence tests:";
    TIMER_START(lengthOfLongestIncreasingSubsequence);
    lengthOfLongestIncreasingSubsequence_scaffold("[10, 9, 2, 5, 3, 7, 101, 18]", 4);
    lengthOfLongestIncreasingSubsequence_scaffold("[10, 9, 2, 5, 3, 4, 7, 101, 18]", 5);
    TIMER_STOP(lengthOfLongestIncreasingSubsequence);
    util::Log(logESSENTIAL) << "lengthOfLongestIncreasingSubsequence using " << TIMER_MSEC(lengthOfLongestIncreasingSubsequence) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findNumberOfLongestIncreasingSubsequence tests:";
    TIMER_START(findNumberOfLongestIncreasingSubsequence);
    findNumberOfLongestIncreasingSubsequence_scaffold("[10, 9, 2, 5, 3, 7, 101, 18]", 4);
    findNumberOfLongestIncreasingSubsequence_scaffold("[2,2,2,2,2]", 5);
    findNumberOfLongestIncreasingSubsequence_scaffold("[1,3,5,4,7]", 2);
    TIMER_STOP(findNumberOfLongestIncreasingSubsequence);
    util::Log(logESSENTIAL) << "findNumberOfLongestIncreasingSubsequence using " << TIMER_MSEC(findNumberOfLongestIncreasingSubsequence) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findLengthOfLongestContinuousIncreasingSubsequence tests:";
    TIMER_START(findLengthOfLongestContinuousIncreasingSubsequence);
    findLengthOfLCIS_scaffold("[2,2,2,2,2]", 1);
    findLengthOfLCIS_scaffold("[1,3,5,4,7]", 3);
    TIMER_STOP(findLengthOfLongestContinuousIncreasingSubsequence);
    util::Log(logESSENTIAL) << "findLengthOfLongestContinuousIncreasingSubsequence using " << TIMER_MSEC(findLengthOfLongestContinuousIncreasingSubsequence) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestConsecutive tests:";
    TIMER_START(longestConsecutive);
    longestConsecutive_scaffold("[2,2,2,2,2]", 1);
    longestConsecutive_scaffold("[1,3,5,4,7]", 3);
    longestConsecutive_scaffold("[100,4,200,1,3,2]", 4);
    longestConsecutive_scaffold("[-4,-1,4,-5,1,-6,9,-6,0,2,2,7,0,9,-3,8,9,-2,-6,5,0,3,4,-2]", 12);
    longestConsecutive_scaffold("[0,3,7,2,5,8,4,6,0,1]", 9);
    TIMER_STOP(longestConsecutive);
    util::Log(logESSENTIAL) << "longestConsecutive using " << TIMER_MSEC(longestConsecutive) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestStrChain tests:";
    TIMER_START(longestStrChain);
    longestStrChain_scaffold("[a,b,ba,bca,bda,bdca]", 4);
    longestStrChain_scaffold("[xbc,pcxbcf,xb,cxbc,pcxbc]", 5);
    longestStrChain_scaffold("[abcd,dbqca]", 1);
    TIMER_STOP(longestStrChain);
    util::Log(logESSENTIAL) << "longestStrChain using " << TIMER_MSEC(longestStrChain) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestSubsequence tests:";
    TIMER_START(longestSubsequence);
    longestSubsequence_scaffold("[1,2,3,4]", 1, 4);
    longestSubsequence_scaffold("[1,3,5,7]", 1, 1);
    longestSubsequence_scaffold("[1,5,7,8,5,3,4,2,1]", -2, 4);
    TIMER_STOP(longestSubsequence);
    util::Log(logESSENTIAL) << "longestSubsequence using " << TIMER_MSEC(longestSubsequence) << " milliseconds";
}
