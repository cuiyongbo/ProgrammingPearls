#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 295, 480, 239 */

namespace stray_dog {
class MedianFinder {
/*
    Find Median from Data Stream(the input stream may be unsorted) O(logn) + O(1)
    Median is the middle value in an ordered integer list. If the size of the list is even, 
    there is no middle value. So the median is the mean of the two middle value.

    Design a data structure that supports the following two operations:
        void addNum(int num) – Add a integer number from the data stream to the data structure.
        double findMedian() – Return the median of all elements so far.

    Related questions: 
        Sliding Window Median
        Finding MK Average
        Sequentially Ordinal Rank Tracker
*/
public:
    void addNum(int num);
    double findMedian();

private:
    priority_queue<int, vector<int>, less<int>> m_max_pq;
    priority_queue<int, vector<int>, greater<int>> m_min_pq;
};

void MedianFinder::addNum(int num) {
    m_max_pq.push(num);
    m_min_pq.push(m_max_pq.top());
    m_max_pq.pop();
    if (m_min_pq.size() > m_max_pq.size()) {
        m_max_pq.push(m_min_pq.top());
        m_min_pq.pop();
    }
}

double MedianFinder::findMedian() {
    if (m_max_pq.size() > m_min_pq.size()) {
        return m_max_pq.top();
    } else {
        return (m_max_pq.top() + m_min_pq.top()) * 0.5;
    }
}

}

void MedianFinder_scaffold(string operations, string args, string expectedOutputs) {
    util::Log(logESSENTIAL) << "case(" << operations << ") begin:";
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    stray_dog::MedianFinder tm;
    int n = operations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "addNum") {
            tm.addNum(std::stoi(funcArgs[i][0]));
        } else if (funcOperations[i] == "findMedian") {
            double actual = tm.findMedian();
            if (actual != std::stod(ans[i])) {
                util::Log(logERROR) << "findMedian failed, Expected: " << ans[i] << ", actual: " << actual;
            } else {
                util::Log(logINFO) << "findMedian passed";
            }
        }
    }
}

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k);
    vector<int> maxSlidingWindow(vector<int>& nums, int k);
};

vector<double> Solution::medianSlidingWindow(vector<int>& nums, int k) {
/*
    Median is the middle value in an ordered integer list. 
    If the size of the list is even, there is no middle value. 
    So the median is the mean of the two middle value.

    Given an unsorted array nums, there is a sliding window of size k 
    which is moving from the very left of the array to the very right. 
    You can only see the k numbers in the window. Each time the sliding window 
    moves right by one position. Your job is to output the median array 
    for each window in the original array.

    related exercises:
        Minimize Malware Spread II
        Largest Unique Number
        Partition Array According to Given Pivot
*/
    int sz = nums.size();
    int step = (k%2==1) ? 0 : 1;
    vector<double> ans; ans.reserve(sz-k+1);
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for (int i=k; i<=sz; ++i) {
        auto m1 = std::next(windows.begin(), (k-1)/2);
        auto m2 = std::next(m1, step);
        ans.push_back((*m1+*m2)*0.5);
        if (i<sz) {
            windows.insert(nums[i]);
            windows.erase(windows.lower_bound(nums[i-k]));
        }
    }
    return ans;
}

vector<int> Solution::maxSlidingWindow(vector<int>& nums, int k) {
/*
    Given an array nums, there is a sliding window of size k which is moving 
    from the very left of the array to the very right. You can only see the k 
    numbers in the window. Each time the sliding window moves right by one position.

    For example, Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

        Window position                Max
        ---------------               -----
        [1  3  -1] -3  5  3  6  7       3
         1 [3  -1  -3] 5  3  6  7       3
         1  3 [-1  -3  5] 3  6  7       5
         1  3  -1 [-3  5  3] 6  7       5
         1  3  -1  -3 [5  3  6] 7       6
         1  3  -1  -3  5 [3  6  7]      7

    Therefore, return the max sliding window as [3,3,5,5,6,7].
    Note: You may assume k is always valid, ie: 1 ≤ k ≤ input array’s size for non-empty array.
*/

if (0) { // binary tree solution
    int sz = nums.size();
    vector<int> ans; ans.reserve(sz-k+1);
    multiset<int> windows(nums.begin(), nums.begin()+k);
    for (int i=k; i<=sz; ++i) {
        ans.push_back(*(windows.rbegin()));
        if (i<sz) {
            windows.insert(nums[i]);
            windows.erase(windows.lower_bound(nums[i-k]));
        }
    }
    return ans;    
}

{ // priority_queue solution
    auto cmp = [&] (int l, int r) { return nums[l] < nums[r];};
    priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
    for (int i=0; i<k; ++i) {
        pq.push(i);
    }
    int sz = nums.size();
    vector<int> ans; ans.reserve(sz-k+1);
    for (int i=k; i<sz; ++i) {
        while (pq.top() < i-k) {
            pq.pop();
        }
        ans.push_back(nums[pq.top()]);
        pq.push(i);
    }
    ans.push_back(nums[pq.top()]);
    return ans;
}

}

void maxSlidingWindow_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.maxSlidingWindow(nums, input2);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual:" << numberVectorToString<int>(actual);
    }
}

void medianSlidingWindow_scaffold(string input1, int input2, string expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    vector<double> expected = stringTo1DArray<double>(expectedResult);
    vector<double> actual = ss.medianSlidingWindow(nums, input2);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual:" << numberVectorToString(actual);
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running MedianFinder tests:";
    TIMER_START(MedianFinder);
    MedianFinder_scaffold("[MedianFinder,addNum,findMedian,addNum,findMedian,addNum,findMedian]", 
                    "[[],[1],[],[3],[],[2],[]]",
                    "[null,null,1,null,2,null,2]");
    MedianFinder_scaffold("[MedianFinder,addNum,findMedian,addNum,findMedian,addNum,findMedian,addNum,findMedian,addNum,findMedian]", 
                    "[[],[-1],[],[-2],[],[-3],[],[-4],[],[-5],[]]",
                    "[null,null,-1.00000,null,-1.50000,null,-2.00000,null,-2.50000,null,-3.00000]");
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
