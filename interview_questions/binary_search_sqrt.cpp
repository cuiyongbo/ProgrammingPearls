#include "leetcode.h"
#include "util/trip_farthest_insertion.hpp"

using namespace std;
using namespace osrm;

/* leetcode exercises: 69 */

class Solution
{
public:
    int mySqrt(int a);
};

int Solution::mySqrt(int a)
{
    /*
        Implement ``int sqrt(int x)``.
        Compute and return the square root of x.
        x is guaranteed to be a non-negative integer.

        since the return type is an integer, the decimal digits 
        are truncated and only the integer part of the result is returned.
        
        Hint: lower bound
    */

    BOOST_ASSERT_MSG(a>=0, "a must be non-negative");

    long left = 0;
    long right = long(a)+1;
    while(left < right)
    {
        long mid = left + (right-left)/2;

        if(mid * mid < a)
        {
            left = mid+1;
        }
        else
        {
            right = mid;
        }
    }
    return left*left > a ? left-1 : left;
}

void mySqrt_scaffold(int input, int expectedResult)
{
    Solution ss;
    int actual = ss.mySqrt(input);
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

    util::Log(logESSENTIAL) << "Running mySqrt tests:";
    TIMER_START(mySqrt);
    mySqrt_scaffold(0, 0);
    mySqrt_scaffold(1, 1);
    mySqrt_scaffold(4, 2);
    mySqrt_scaffold(8, 2);
    TIMER_STOP(mySqrt);
    util::Log(logESSENTIAL) << "mySqrt using " << TIMER_MSEC(mySqrt) << " milliseconds";
}
