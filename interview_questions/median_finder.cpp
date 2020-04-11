#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 295, 480, 239 */

class MedianFinder 
{
/*
    Find Median from Data Stream O(logn) + O(1)
    Median is the middle value in an ordered integer list. If the size of the list is even, 
    there is no middle value. So the median is the mean of the two middle value.

    Design a data structure that supports the following two operations:
        void addNum(int num) – Add a integer number from the data stream to the data structure.
        double findMedian() – Return the median of all elements so far.
*/
using NodeIter = vector<int>::iterator;
public:
    MedianFinder() = default;
    void addNum(int num);
    double findMedian();

private:
    vector<int> m_data;
};

void MedianFinder::addNum(int num)
{
    NodeIter it = m_data.begin();
    for(; it != m_data.end(); ++it)
    {
        if(*it >= num) break;
    }
    m_data.insert(it, num);
}

double MedianFinder::findMedian()
{
    BOOST_ASSERT(!m_data.empty());

    size_t s = m_data.size();
    if(s%2 == 0)
    {
        return (m_data[s/2] + m_data[s/2-1]) * 0.5;
    }
    else
    {
        return m_data[s/2];
    }
}

void MedianFinder_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = toStringArray(operations);
    vector<vector<string>> funcArgs = to2DStringArray(args);
    vector<string> ans = toStringArray(expectedOutputs);
    MedianFinder tm;
    int n = (int)operations.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "addNum")
        {
            tm.addNum(std::stoi(funcArgs[i][0]));
        }
        else if(funcOperations[i] == "findMedian")
        {
            double actual = tm.findMedian();
            if(actual != std::stod(ans[i]))
            {
                util::Log(logERROR) << "findMedian failed, Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << "findMedian passed";
            }
        }
    }
}

class Solution 
{
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k);
    vector<int> maxSlidingWindow(vector<int>& nums, int k);
};

vector<double> Solution::medianSlidingWindow(vector<int>& nums, int k)
{
    /*
        Median is the middle value in an ordered integer list. 
        If the size of the list is even, there is no middle value. 
        So the median is the mean of the two middle value.

        Given an array nums, there is a sliding window of size k 
        which is moving from the very left of the array to the very right. 
        You can only see the k numbers in the window. Each time the sliding window 
        moves right by one position. Your job is to output the median array 
        for each window in the original array.
    */

    int size = (int)nums.size();
    BOOST_ASSERT_MSG(k<=size, "the size of sliding window must not be larger than the number of input array");

    vector<double> ans(size-k+1);
    const int step = int((k+1) & 1);
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for(int i=k; i<=size; ++i)
    {
        auto m1 = std::next(windows.begin(), (k-1)/2);
        ans[i-k] = (*m1 + *(std::next(m1, step))) * 0.5;

        windows.emplace(nums[i]);
        windows.erase(windows.lower_bound(nums[i-k]));
    }
    return ans;
}

vector<int> Solution::maxSlidingWindow(vector<int>& nums, int k)
{
    /*
        Given an array nums, there is a sliding window of size k 
        which is moving from the very left of the array to the very right. 
        You can only see the k numbers in the window. Each time the sliding window 
        moves right by one position. Return the max sliding window.
    */

    int size = (int)nums.size();
    BOOST_ASSERT_MSG(k<=size, "the size of sliding window must not be larger than the number of input array");

    vector<int> ans(size-k+1);
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for(int i=k; i<=size; ++i)
    {
        ans[i-k] = *(std::prev(windows.end()));

        windows.emplace(nums[i]);
        windows.erase(windows.lower_bound(nums[i-k]));
    }
    return ans;
}

void medianSlidingWindow_scaffold(string input1, int input2, string expectedResult)
{
    Solution ss;
    vector<int> nums = stringToIntegerVector(input1);
    vector<double> expected = stringToDoubleVector(expectedResult);
    vector<double> actual = ss.medianSlidingWindow(nums, input2);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << numberVectorToString(actual);
    }
}

void maxSlidingWindow_scaffold(string input1, int input2, string expectedResult)
{
    Solution ss;
    vector<int> nums = stringToIntegerVector(input1);
    vector<int> expected = stringToIntegerVector(expectedResult);
    vector<int> actual = ss.maxSlidingWindow(nums, input2);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << numberVectorToString(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running MedianFinder tests:";
    TIMER_START(MedianFinder);
    MedianFinder_scaffold("[MedianFinder,addNum,findMedian,addNum,findMedian,addNum,findMedian]", 
                    "[[],[1],[],[3],[],[2],[]]",
                    "[null,null,1,null,2,null,2]");
    TIMER_STOP(MedianFinder);
    util::Log(logESSENTIAL) << "MedianFinder using " << TIMER_MSEC(MedianFinder) << " milliseconds";

    util::Log(logESSENTIAL) << "Running medianSlidingWindow tests:";
    TIMER_START(medianSlidingWindow);
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 3, "[1,-1,-1,3,5,6]");
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 4, "[0,1,1,4,5.5]");
    medianSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 5, "[1,3,3,5]");
    TIMER_STOP(medianSlidingWindow);
    util::Log(logESSENTIAL) << "medianSlidingWindow using " << TIMER_MSEC(medianSlidingWindow) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxSlidingWindow tests:";
    TIMER_START(maxSlidingWindow);
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 3, "[3,3,5,5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 4, "[3,5,5,6,7]");
    maxSlidingWindow_scaffold("[1,3,-1,-3,5,3,6,7]", 5, "[5,5,6,7]");
    TIMER_STOP(maxSlidingWindow);
    util::Log(logESSENTIAL) << "maxSlidingWindow using " << TIMER_MSEC(maxSlidingWindow) << " milliseconds";
}
