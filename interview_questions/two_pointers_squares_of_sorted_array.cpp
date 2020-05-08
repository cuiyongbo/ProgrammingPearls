#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 977 */

class Solution 
{
public:
    vector<int> sortedSquares(vector<int>& A);
    int subarraysWithKDistinct(vector<int>& A, int K);
    int preimageSizeFZF(int K);

private:
    int subarraysWithKDistinct_bruteforce(vector<int>& A, int K);
    int subarraysWithKDistinct_triky(vector<int>& A, int K);

};

vector<int> Solution::sortedSquares(vector<int>& A)
{
    /*
        Given an array of integers A sorted in non-decreasing order, 
        return an array of the squares of each number, 
        also in sorted non-decreasing order.
    */

    auto mid = std::lower_bound(A.begin(), A.end(), 0);

    deque<int> left;
    for(auto it=A.begin(); it != mid; ++it)
    {
        left.push_front(*it * *it);
    }

    deque<int> right;
    for(auto it=mid; it != A.end(); ++it)
    {
        right.push_back(*it * *it);
    }

    vector<int> ans(A.size());
    int ls = left.size();
    int rs = right.size();
    int i=0, j=0, k=0;
    while(i<ls || j<rs)
    {
        if(j==rs || (i<ls && left[i] < right[j]))
        {
            ans[k++] = left[i++];
        }
        else
        {
            ans[k++] = right[j++];
        }
    }
    return ans;
}

int Solution::subarraysWithKDistinct(vector<int>& A, int K)
{
    /*
        Given an array A of positive integers, call a (contiguous, not necessarily distinct) 
        subarray of A good if the number of different integers in that subarray is exactly K.
        (For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.)
        Return the number of good subarrays of A.

        Note:

            1 <= A.length <= 20000
            1 <= A[i] <= A.length
            1 <= K <= A.length
    */

    // return subarraysWithKDistinct_bruteforce(A, K);
    return subarraysWithKDistinct_triky(A, K);
}

int Solution::subarraysWithKDistinct_triky(vector<int>& A, int K)
{
    // f(x) means subarray count with x or less than x distinct integer(s)
    // so ans = f(k) - f(k-1)

    int len = (int)A.size();
    auto subarrayCount = [&](int k)
    {
        int j = 0;
        int ans = 0;
        vector<int> count(len + 1); // 1 <= A[i] <= A.length
        for(int i=0; i<len; i++)
        {
            // a new distinct integer coming
            if(count[A[i]]++ == 0) --k;

            // k<0 means more than k distinct integers have been found
            while(k < 0)
            {
                // remove a distinct integer
                if(--count[A[j++]] == 0) ++k;
            }
            ans += i-j+1;
        }
        return ans;
    };

    return subarrayCount(K) - subarrayCount(K-1);
}

int Solution::subarraysWithKDistinct_bruteforce(vector<int>& A, int K)
{
    set<int> s;
    int ans = 0;
    int len = (int)A.size();
    for(int i=0; i<len-K+1; i++)
    {
        s.clear();
        for(int j=i; j<len; j++)
        {
            s.insert(A[j]);
            if((int)s.size() > K)
                break;
            else if((int)s.size() == K)
                ans++;
        }
    }
    return ans;
}

int Solution::preimageSizeFZF(int K)
{
    /*
        Let f(x) be the number of zeroes at the end of x!. 
        (Recall that x! = 1 * 2 * 3 * ... * x, and by convention, 0! = 1.)

        For example, f(3) = 0 because 3! = 6 has no zeroes at the end, 
        while f(11) = 2 because 11! = 39916800 has 2 zeroes at the end. 
        Given K, find how many non-negative integers x have the property that f(x) = K.

        Example 1:
        Input: K = 0
        Output: 5
        Explanation: 0!, 1!, 2!, 3!, and 4! end with K = 0 zeroes.

        Example 2:
        Input: K = 5
        Output: 0
        Explanation: There is no x such that x! ends in K = 5 zeroes.

        Note: K will be an integer in the range [0, 10^9].
    */

   // g(k) means the count of number with k or less than k trailing 0s at the end.
   // so ans = g(k) - g(k-1)

   auto g = [&](int k)
   {
       int ans = 0;
       return ans;
   };

    return g(K) - g(K-1);
}

void sortedSquares_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> nums = stringTo1DArray<int>(input);
    vector<int> actual = ss.sortedSquares(nums);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input  << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << numberVectorToString(actual);
    }
}

void subarraysWithKDistinct_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.subarraysWithKDistinct(nums, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2  << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void preimageSizeFZF_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.preimageSizeFZF(input);
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

    util::Log(logESSENTIAL) << "Running sortedSquares tests: ";
    TIMER_START(sortedSquares);
    sortedSquares_scaffold("[-4,-1,0,3,10]", "[0,1,9,16,100]");
    sortedSquares_scaffold("[-7,-3,2,3,11]", "[4,9,9,49,121]");
    TIMER_STOP(sortedSquares);
    util::Log(logESSENTIAL) << "sortedSquares using " << TIMER_MSEC(sortedSquares) << " milliseconds";

    util::Log(logESSENTIAL) << "Running subarraysWithKDistinct tests: ";
    TIMER_START(subarraysWithKDistinct);
    subarraysWithKDistinct_scaffold("[1,2,1,2,3]", 2, 7);
    subarraysWithKDistinct_scaffold("[1,2,1,3,4]", 3, 3);
    subarraysWithKDistinct_scaffold("[1,2,3,4]", 3, 2);
    TIMER_STOP(subarraysWithKDistinct);
    util::Log(logESSENTIAL) << "subarraysWithKDistinct using " << TIMER_MSEC(subarraysWithKDistinct) << " milliseconds";

    util::Log(logESSENTIAL) << "Running preimageSizeFZF tests: ";
    TIMER_START(preimageSizeFZF);
    preimageSizeFZF_scaffold(0, 5);
    preimageSizeFZF_scaffold(5, 0);
    TIMER_STOP(preimageSizeFZF);
    util::Log(logESSENTIAL) << "preimageSizeFZF using " << TIMER_MSEC(preimageSizeFZF) << " milliseconds";

}
