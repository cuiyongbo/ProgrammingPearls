#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 239 */

class Solution
{
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k);

private:
    vector<int> maxSlidingWindow_pq(vector<int>& nums, int k);
};

vector<int> Solution::maxSlidingWindow(vector<int>& nums, int k)
{
    /*
        Given an array nums, there is a sliding window of size k which is moving 
        from the very left of the array to the very right. You can only see the k 
        numbers in the window. Each time the sliding window moves right by one position.

        For example,
        Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

            Window position                Max
            ---------------               -----
            [1  3  -1] -3  5  3  6  7       3
             1 [3  -1  -3] 5  3  6  7       3
             1  3 [-1  -3  5] 3  6  7       5
             1  3  -1 [-3  5  3] 6  7       5
             1  3  -1  -3 [5  3  6] 7       6
             1  3  -1  -3  5 [3  6  7]      7
        
        Therefore, return the max sliding window as [3,3,5,5,6,7].

        Note:  You may assume k is always valid, ie: 1 ≤ k ≤ input array’s size for non-empty array.
    */

    return maxSlidingWindow_pq(nums, k);    
}

vector<int> Solution::maxSlidingWindow_pq(vector<int>& nums, int k)
{
    vector<int> ans;
    priority_queue<pair<int,int>, vector<pair<int,int>>> pq;
    for(int i=0; i<k; i++)
    {
        pq.push({nums[i], i});
    }

    for(int i=k; i<(int)nums.size(); i++)
    {
        while(pq.top().second < i-k)
        {
            pq.pop();
        }

        ans.push_back(pq.top().first);
        pq.push({nums[i], i});
    }

    ans.push_back(pq.top().first);
    
    return ans;
}

void maxSlidingWindow_scaffold(string input1, int input2, string expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.maxSlidingWindow(nums, input2);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << numberVectorToString<int>(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running maxSlidingWindow tests:";
    TIMER_START(maxSlidingWindow);
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 3, "[3,3,5,5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 6, "[5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 8, "[7]");
    TIMER_STOP(maxSlidingWindow);
    util::Log(logESSENTIAL) << "maxSlidingWindow using " << TIMER_MSEC(maxSlidingWindow) << " milliseconds";
}
