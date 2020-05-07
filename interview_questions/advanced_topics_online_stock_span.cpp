#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 901, 907, 1019 */

/*
    Write a class StockSpanner which collects daily price quotes for some stock, 
    and returns the span of that stock’s price for the current day.

    The span of the stock’s price today is defined as the maximum number 
    of consecutive days (starting from today and going backwards) for which 
    the price of the stock was less than or equal to today’s price.

    For example, if the price of a stock over the next 7 days were [100, 80, 60, 70, 60, 75, 85], 
    then the stock spans would be [1, 1, 1, 2, 1, 4, 6].

    For example, Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
    Output: [null,1,1,1,2,1,4,6]
    Explanation: 
        First, S = StockSpanner() is initialized.  Then:
            S.next(100) is called and returns 1,
            S.next(80) is called and returns 1,
            S.next(60) is called and returns 1,
            S.next(70) is called and returns 2,
            S.next(60) is called and returns 1,
            S.next(75) is called and returns 4,
            S.next(85) is called and returns 6.

    Note that (for example) S.next(75) returned 4, because the last 4 prices
    (including today's price of 75) were less than or equal to today's price.
    Note:
        Calls to StockSpanner.next(int price) will have 1 <= price <= 10^5.
        There will be at most 10000 calls to StockSpanner.next per test case.
        There will be at most 150000 calls to StockSpanner.next across all test cases.
        The total time limit for this problem has been reduced by 75% for C++, and 50% for all other languages.
*/

class StockSpanner 
{
public:
    StockSpanner() {}
    int next(int price);

private:
    int next_dp(int price);
    int next_monotonic_stack(int price);

private:
    stack<pair<int, int>> m_st;
    vector<int> m_span;
    vector<int> m_prices;  
};

int StockSpanner::next(int price)
{
    return next_monotonic_stack(price);
}

int StockSpanner::next_monotonic_stack(int price)
{
    auto p = std::make_pair(price, 1);
    while(!m_st.empty() && m_st.top().first <= price)
    {
        p.second += m_st.top().second;
        m_st.pop();
    }
    m_st.push(p);
    return p.second;
}

int StockSpanner::next_dp(int price)
{
    // m_span[i] means span of price i;
    int j = m_prices.size();

    m_prices.push_back(price);
    m_span.push_back(1);

    int i=j;
    while(i>=0 && m_prices[i] <= price)
        i -= m_span[i];

    return m_span[j] = j-i;
}

void StockSpanner_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    StockSpanner tm;
    int n = (int)ans.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "next")
        {
            int actual = tm.next(std::stoi(funcArgs[i][0]));
            if(actual != std::stoi(ans[i]))
            {
                util::Log(logERROR) << "next(" << funcArgs[i][0] << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << "next(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

class Solution 
{
public:
    int sumSubarrayMins(vector<int>& A);
    vector<int> nextLargerNodes(ListNode* head);

private:
    int sumSubarrayMins_bruteforce(vector<int>& A);
    int sumSubarrayMins_monotonic_stack(vector<int>& A);
};

vector<int> Solution::nextLargerNodes(ListNode* head)
{
    /*
        We are given a linked list with header as the first node.
        Let's number the nodes in the list: node_1, node_2, node_3, ... etc.
        
        Each node may have a next larger value: for node_i, next_larger(node_i) 
        is the node_j.val such that j > i, node_j.val > node_i.val, and j is the
        smallest possible choice.  If such a j does not exist, the next larger value is 0.
        
        Return an array of integers answer, where answer[i] = next_larger(node_{i+1}).
    */

    map<ListNode*, int> m;
    stack<ListNode*> st;
    for(ListNode* p=head; p != NULL; p=p->next)
    {
        while(!st.empty() && st.top()->val < p->val)
        {
            m[st.top()] = p->val;
            st.pop();
        }
        
        st.push(p);
    }

    while(!st.empty())
    {
        m[st.top()] = 0;
        st.pop();
    }

    int i=0;
    vector<int> ans(m.size());
    for(ListNode* p=head; p != NULL; p=p->next)
    {
        ans[i++] = m[p];
    }
    return ans;
}

int Solution::sumSubarrayMins(vector<int>& A)
{
    /*
        Given an array of integers A, find the sum of min(B), where B ranges over every (contiguous) subarray of A.
        Since the answer may be large, return the answer modulo 10^9 + 7.
        Example 1:

            Input: [3,1,2,4]
            Output: 17
            Explanation: Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. 
            Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.  Sum is 17.
    */

    return sumSubarrayMins_monotonic_stack(A);
}

int Solution::sumSubarrayMins_monotonic_stack(vector<int>& A)
{
    bool newRound = false;
    stack<pair<int, int>> st;
    const int MODULO_NUM = 1e9 + 7;
    int size = (int)A.size();
    for(int i=0; i<size; i++)
    {
        newRound = true;
        for(int j=i; j<size; j++)
        {
            if(!newRound && !st.empty() && st.top().first < A[j])
                st.top().second++;
            else
                st.push(std::make_pair(A[j], 1));
            
            newRound = false;
        }
    }

    int ans = 0;
    while(!st.empty())
    {
        auto p = st.top(); st.pop();
        ans = (ans + p.first * p.second) % MODULO_NUM;
    }
    return ans;
}

int Solution::sumSubarrayMins_bruteforce(vector<int>& A)
{
    const int MODULO_NUM = 1e9 + 7;
    int size = (int)A.size();
    int ans = 0;
    for(int i=0; i<size; i++)
    {
        int m = A[i];
        for(int j=i; j<size; j++)
        {
            // fetch minimum in A[i, j] 
            m = std::min(m, A[j]);
            ans = (ans + m)%MODULO_NUM;
        }
    }
    return ans;
}

void sumSubarrayMins_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.sumSubarrayMins(nums);
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

void nextLargerNodes_scaffold(string input, string expectedResult)
{
    Solution ss;
    ListNode* head = stringToListNode(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.nextLargerNodes(head);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << numberVectorToString(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running StockSpanner tests:";
    TIMER_START(StockSpanner);
    StockSpanner_scaffold("[StockSpanner,next,next,next,next,next,next,next]", 
                    "[[],[100],[80],[60],[70],[60],[75],[85]]",
                    "[null,1,1,1,2,1,4,6]");
    TIMER_STOP(StockSpanner);
    util::Log(logESSENTIAL) << "StockSpanner using " << TIMER_MSEC(StockSpanner) << " milliseconds";

    util::Log(logESSENTIAL) << "Running sumSubarrayMins tests:";
    TIMER_START(sumSubarrayMins);
    sumSubarrayMins_scaffold("[3,1,2,4]", 17);
    TIMER_STOP(sumSubarrayMins);
    util::Log(logESSENTIAL) << "sumSubarrayMins using " << TIMER_MSEC(sumSubarrayMins) << " milliseconds";

    util::Log(logESSENTIAL) << "Running nextLargerNodes tests:";
    TIMER_START(nextLargerNodes);
    nextLargerNodes_scaffold("[2,1,5]", "[5,5,0]");
    nextLargerNodes_scaffold("[2,7,4,3,5]", "[7,0,5,5,0]");
    nextLargerNodes_scaffold("[1,7,5,1,9,2,5,1]", "[7,9,9,9,0,5,0,0]");
    nextLargerNodes_scaffold("[5,4,3,2,1]", "[0,0,0,0,0]");
    TIMER_STOP(nextLargerNodes);
    util::Log(logESSENTIAL) << "nextLargerNodes using " << TIMER_MSEC(nextLargerNodes) << " milliseconds";
}
