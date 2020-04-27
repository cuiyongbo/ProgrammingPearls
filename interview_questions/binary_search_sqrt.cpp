#include "leetcode.h"
#include "util/trip_farthest_insertion.hpp"

using namespace std;
using namespace osrm;

/* leetcode exercises: 69, 875, 1011 */

class Solution
{
public:
    int mySqrt(int a);
    int minEatingSpeed(vector<int>& piles, int H);
    int shipWithinDays(vector<int>& weights, int D);
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

int Solution::minEatingSpeed(vector<int>& piles, int H)
{
    /*
        Koko loves to eat bananas. There are N piles of bananas, the i-th pile has piles[i] bananas.  
        The guards have gone and will come back in H hours.

        Koko can decide her bananas-per-hour eating speed of K. Each hour, she chooses some pile of bananas, 
        and eats K bananas from that pile. If the pile has less than K bananas, she eats all of them instead, 
        and won’t eat any more bananas during this hour.

        Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back.
        Return the minimum integer K such that she can eat all the bananas within H hours.
    */

    int pileCount = (int)piles.size();
    BOOST_ASSERT_MSG(H >= pileCount, "H must be not less than the number of piles");

    auto it = std::minmax_element(piles.begin(), piles.end());
    if(H == pileCount)
    {
        return *(it.second);
    }

    auto etaBanana = [&](int speed)
    {
        int h = 0;
        for(auto p: piles)
        {
            h += (p + speed-1)/speed;
        }
        return (h > H) ? -1 : ((h == H) ? 0 : 1);
    };

    int l = *(it.first);
    int r = *(it.second);
    while(l < r)
    {
        int m = l + (r-l)/2;
        if(etaBanana(m) < 0)
        {
            // can't eat up all piles
            l = m+1;
        }
        else
        {
            r = m;
        }
    }

    return l;
}

int Solution::shipWithinDays(vector<int>& weights, int D)
{
    /*
        A conveyor belt has packages that must be shipped from one port to another within D days.

        The i-th package on the conveyor belt has a weight of weights[i]. Each day, we load the ship 
        with packages on the conveyor belt (*in the order given by weights*). We may not load more weight 
        than the maximum weight capacity of the ship and we can't split one package.

        Return the least weight capacity of the ship that will result in all the packages on 
        the conveyor belt being shipped within D days.
    */

    BOOST_ASSERT_MSG(D>0, "D must be non-negative");
    int packageCount = (int)weights.size();
    auto loadCargo = [&] (int s)
    {
        int k=0;
        int days = 0;
        while(k < packageCount)
        {
            int left = s;
            for(; k<packageCount; ++k)
            {
                if(left < weights[k]) break;
                left -= weights[k];
            }
            ++days;
        }
        return days <= D;
    };

    int l = *(std::max_element(weights.begin(), weights.end()));
    int r = std::accumulate(weights.begin(), weights.end(), 0) + 1;
    while(l < r)
    {
        int m = l + (r-l)/2;
        if(!loadCargo(m))
        {
            l = m+1;
        }
        else
        {
            r = m;
        }
    }
    return l;
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

void minEatingSpeed_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> piles = stringTo1DArray<int>(input1);
    int actual = ss.minEatingSpeed(piles, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void shipWithinDays_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> piles = stringTo1DArray<int>(input1);
    int actual = ss.shipWithinDays(piles, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
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

    util::Log(logESSENTIAL) << "Running minEatingSpeed tests:";
    TIMER_START(minEatingSpeed);
    minEatingSpeed_scaffold("[3,6,7,11]", 8, 4);
    minEatingSpeed_scaffold("[30,11,23,4,20]", 5, 30);
    minEatingSpeed_scaffold("[30,11,23,4,20]", 6, 23);
    TIMER_STOP(minEatingSpeed);
    util::Log(logESSENTIAL) << "minEatingSpeed using " << TIMER_MSEC(minEatingSpeed) << " milliseconds";

    util::Log(logESSENTIAL) << "Running shipWithinDays tests:";
    TIMER_START(shipWithinDays);
    shipWithinDays_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, 15);
    shipWithinDays_scaffold("[3,2,2,4,1,4]", 3, 6);
    shipWithinDays_scaffold("[1,2,3,1,1]", 4, 3);
    shipWithinDays_scaffold("[1,2,3,4,5,6,7,8,9,10]", 10, 10);
    TIMER_STOP(shipWithinDays);
    util::Log(logESSENTIAL) << "shipWithinDays using " << TIMER_MSEC(shipWithinDays) << " milliseconds";
}
