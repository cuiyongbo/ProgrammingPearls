#include "leetcode.h"

using namespace std;

class Solution {
public:
    int lengthOfLongestIncreasingSubsequence(vector<int>& nums);
    int findNumberOfLongestIncreasingSubsequence(vector<int>& nums);
    int lengthOfLongestIncreasingSubarray(vector<int>& nums);
};

int Solution::lengthOfLongestIncreasingSubsequence(vector<int>& nums)
{
    if(nums.empty())
        return 0;

    int ans = 1;
    int n = nums.size();

    // dp[i] means lengthOfLongestIncreasingSubsequence ending with nums[i]
    vector<int> dp(n, 1);
    for (int i = 1; i < n; ++i)
    {
        for(int j=0; j<i; j++)
        {
            if (nums[i] > nums[j])
            {
                dp[i] = max(dp[i], dp[j]+1);
                ans = max(ans, dp[i]);
            }
        }
    }
    return ans;
}

int Solution::findNumberOfLongestIncreasingSubsequence(vector<int>& nums)
{
    if(nums.empty())
        return 0;

    int maxLen = 1;
    int n = nums.size();

    // count[i] means numberOfLIS ending with nums[i]
    // length[i] means lengthOfLongestIncreasingSubsequence ending with nums[i]
    vector<int> count(n, 1);
    vector<int> length(n, 1);
    for (int i = 1; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if(nums[i] > nums[j])
            {
                if(length[j] + 1 > length[i])
                {
                    length[i] = length[j] + 1;
                    count[i] = count[j];
                }
                else if(length[j]+1 == length[i])
                {
                    count[i] += count[j];
                }
            }
        }
        maxLen = max(maxLen, length[i]);
    }

    //cout << "lengthOfLongestIncreasingSubsequence: " << ans << "\n";

    int ans = 0;
    for (int i = 0; i < n; ++i)
    {
        if(length[i] == maxLen)
            ans += count[i];
    }
    return ans;
}

int Solution::lengthOfLongestIncreasingSubarray(vector<int>& nums)
{
    if(nums.empty())
        return 0;

    int ans = 1;
    int n = nums.size();

    // dp[i] means LengthOfLongestIncreasingSubarray ending with nums[i]
    vector<int> dp(n, 1);
    for(int i=1; i<n; i++)
    {
        if(nums[i] > nums[i-1])
        {
            dp[i] = max(dp[i], dp[i-1] + 1);
            ans = max(ans, dp[i]);
        }
    }
    return ans;
}

int main()
{
    vector<int> input {10,9,2,5,3,7,101,18};
    printVector(input);

    Solution ss;
    cout << "lengthOfLongestIncreasingSubsequence: " << ss.lengthOfLongestIncreasingSubsequence(input) << "\n";
    cout << "NumOfLIS: " << ss.findNumberOfLongestIncreasingSubsequence(input) << "\n";
    return 0;
}
