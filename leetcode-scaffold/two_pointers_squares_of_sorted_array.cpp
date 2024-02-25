#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 977, 992, 172, 793*/

class Solution {
public:
    vector<int> sortedSquares(vector<int>& A);
    int subarraysWithKDistinct(vector<int>& A, int K);
    int trailingZeroes(int n);
    int preimageSizeFZF(int K);
};

vector<int> Solution::sortedSquares(vector<int>& A) {
/*
    Given an array of integers A sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.
*/
    int sz = A.size();
    int l = 0;
    int r = sz;
    while (l < r) {
        int m = (l+r)/2;
        if (A[m] < 0) {
            l = m+1;
        } else {
            r = m;
        }
    }

    int i=l-1;
    int j = l;
    int k = 0;
    vector<int> ans(sz);
    while (i>=0 || j<sz) {
        if (j==sz || (i>=0 && A[i]*A[i]<A[j]*A[j])) {
            ans[k++] = A[i]*A[i];
            --i;
        } else {
            ans[k++] = A[j]*A[j];
            ++j;
        }
    }
    return ans;
}

int Solution::subarraysWithKDistinct(vector<int>& num, int K) {
/*
    Given an integer array nums and an integer k, return the number of good subarrays of nums.
    A good array is an array where the number of different integers in that array is exactly k.
    For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3. A subarray is a contiguous part of an array.
    Constraints:
        1 <= nums[i], k <= nums.length
    Relative exercises:
        Longest Substring with At Most Two Distinct Characters
        Longest Substring with At Most K Distinct Characters
        Count Vowel Substrings of a String
        Number of Unique Flavors After Sharing K Candies
*/

{ // naive solution take O(n^2) time
}

{ // two-cursor solution
    // worker(k) means the number of subarrays with x or less than x distinct integer(s),
    // so the answer is worker(k)-worker(k-1)
    auto worker = [&] (int k) {
        int ans = 0;
        int j = 0;
        int sz = num.size();
        vector<int> count(sz+1, 0); // 1 <= nums[i] <= nums.length
        for (int i=0; i<sz; ++i) {
            if (count[num[i]] == 0) { // we encounter a new distinct integer
                --k;
            }
            count[num[i]]++;
            while (k<0) {
                --count[num[j]];
                if (count[num[j]] == 0) { // remove a distinct integer
                    ++k;
                }
                ++j;
            }
            ans += (i-j+1);
        }
        return ans;
    };
    return worker(K) - worker(K-1);
}

}

int Solution::trailingZeroes(int n) {
/*
    Given an integer n, return the number of trailing zeroes in n!.
    Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.
*/
    int ans = 0;
    for (; n>0; n/=5) {
        ans += n/5;
    }
    return ans;
}

int Solution::preimageSizeFZF(int K) {
/*
    Let f(x) be the number of zeroes at the end of x!. (Recall that x! = 1 * 2 * 3 * ... * x, and by convention, 0! = 1.)
    For example, f(3) = 0 because 3! = 6 has no zeroes at the end, while f(11) = 2 because 11! = 39916800 has 2 zeroes at the end. 
    Given K, find how many non-negative integers x have the property that f(x) = K.
    Example 1:
        Input: K = 0
        Output: 5
        Explanation: 0!, 1!, 2!, 3!, and 4! end with K = 0 zeroes.
    Example 2:
        Input: K = 5
        Output: 0
        Explanation: There is no x such that x! ends in K = 5 zeroes.
    Hint: https://www.cnblogs.com/grandyang/p/9214055.html
*/
    auto numOfTrailingZeros = [&] (long m) {
        long res = 0;
        for (; m>0; m/=5) {
            res += m/5;
        }
        return res;
    };
    long l = 0;
    long r = 5L * (K+1);
    while (l < r) {
        long m = (l+r)/2;
        long cnt = numOfTrailingZeros(m);
        if (cnt == K) {
            return 5;
        } else if (cnt < K) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return 0;
}

void sortedSquares_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> expected = stringTo1DArray<int>(expectedResult);
    vector<int> nums = stringTo1DArray<int>(input);
    vector<int> actual = ss.sortedSquares(nums);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input  << ", expectedResult: " << expectedResult << ") failed, actual: " << numberVectorToString(actual);
    }
}

void subarraysWithKDistinct_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input1);
    int actual = ss.subarraysWithKDistinct(nums, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2  << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void trailingZeroes_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.trailingZeroes(input);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed， actual: " << actual;
    }
}

void preimageSizeFZF_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.preimageSizeFZF(input);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed， actual: " << actual;
    }
}

int main() {
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

    util::Log(logESSENTIAL) << "Running trailingZeroes tests: ";
    TIMER_START(trailingZeroes);
    trailingZeroes_scaffold(0, 0);
    trailingZeroes_scaffold(5, 1);
    trailingZeroes_scaffold(3, 0);
    trailingZeroes_scaffold(10, 2);
    TIMER_STOP(trailingZeroes);
    util::Log(logESSENTIAL) << "trailingZeroes using " << TIMER_MSEC(trailingZeroes) << " milliseconds";

    util::Log(logESSENTIAL) << "Running preimageSizeFZF tests: ";
    TIMER_START(preimageSizeFZF);
    preimageSizeFZF_scaffold(0, 5);
    preimageSizeFZF_scaffold(5, 0);
    preimageSizeFZF_scaffold(3, 5);
    preimageSizeFZF_scaffold(1000000000, 5);
    TIMER_STOP(preimageSizeFZF);
    util::Log(logESSENTIAL) << "preimageSizeFZF using " << TIMER_MSEC(preimageSizeFZF) << " milliseconds";
}
