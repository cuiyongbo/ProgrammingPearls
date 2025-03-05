#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 69, 875, 1011 */

class Solution {
public:
    int mySqrt(int a);
    int minEatingSpeed(std::vector<int>& piles, int H);
    int shipWithinDays(std::vector<int>& weights, int D);
};


/*
    Implement ``int sqrt(int x)``. Compute and return the square root of x. x is guaranteed to be a non-negative integer.
    since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
    Hint: perform lower_bound search to find the first integer k such than k*k >= x
*/
int Solution::mySqrt(int x) {
    assert(x>=0);
    // perform lower_bound search to find the least integer k that satisfy k*k >=x
    // to prevent intermediates from range overflow, we have to use long type for l, r, m
    long l=0;
    long r=x+1; // r is not inclusive
    while (l < r) {
        long m = (l+r)/2;
        if (m*m < x) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l*l>x ? (l-1) : l;
}


/*
    Koko loves to eat bananas. There are N piles of bananas, the i-th pile has piles[i] bananas. The guards have gone and will come back in H hours.
    Koko can decide her bananas-per-hour eating speed of K. Each hour she chooses some pile of bananas, and eats K bananas from that pile. If the pile has less than K bananas, she eats all of them instead, and won’t eat any more bananas during this hour.
    Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back. Return the minimum integer K such that she can eat all the bananas within H hours. you may assume that piles.length<=H.
*/
int Solution::minEatingSpeed(std::vector<int>& piles, int H) {
    //Hint: perform lower_bound search to find the minimum K which satisfies the requirement
    assert(piles.size()<=H);
    auto can_koko_finish_eating = [&] (int speed) {
        int hours = 0;
        for (auto p: piles) {
            hours += (p+speed-1)/speed;
            if (hours > H) {
                return false;
            }
        }
        return true;
    };
    int l = 1;
    int r = *(std::max_element(piles.begin(), piles.end())) + 1; // r is not inclusive
    while (l < r) {
        int m = (l+r)/2;
        if (!can_koko_finish_eating(m)) { // increase eating speed if koko can't finish eating in H hours
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


/*
    A conveyor belt has packages that must be shipped from one port to another within D days.
    The i-th package on the conveyor belt has a weight of weights[i]. Each day we load the ship with packages on the conveyor belt (*in the order given by weights*). 
    We may not load more weight than the maximum weight capacity of the ship and we can't split one package.
    Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within D days.
    Constraints:
        1 <= D <= weights.length <= 5 * 104
        1 <= weights[i] <= 500
*/
int Solution::shipWithinDays(std::vector<int>& weights, int D) {
    // use lower_bound search to find the least weight capacity of the ship that satisfies the condition
    auto can_finish_order = [&] (int capacity) {
        int days = 0;
        for (int i=0; i<weights.size();) {
            int sum = 0;
            while (i<weights.size() && sum+weights[i]<=capacity) {
                sum += weights[i];
                i++;
            }
            days++;
            if (days > D) {
                return false;
            }
        }
        return true;
    };
    int l = *(std::max_element(weights.begin(), weights.end()));
    //int r = std::accumulate(weights.begin(), weights.end(), 1);
    int r = l*D + 1; // nice catcha
    while (l < r) {
        int m = (l+r)/2;
        if (!can_finish_order(m)) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


void mySqrt_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.mySqrt(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void minEatingSpeed_scaffold(std::string input1, int input2, int expectedResult) {
    Solution ss;
    std::vector<int> piles = stringTo1DArray<int>(input1);
    int actual = ss.minEatingSpeed(piles, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


void shipWithinDays_scaffold(std::string input1, int input2, int expectedResult) {
    Solution ss;
    std::vector<int> piles = stringTo1DArray<int>(input1);
    int actual = ss.shipWithinDays(piles, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running mySqrt tests:");
    TIMER_START(mySqrt);
    mySqrt_scaffold(0, 0);
    mySqrt_scaffold(1, 1);
    mySqrt_scaffold(4, 2);
    mySqrt_scaffold(8, 2);
    mySqrt_scaffold(26, 5);
    mySqrt_scaffold(INT16_MAX, 181);
    mySqrt_scaffold(INT32_MAX, 46340);
    TIMER_STOP(mySqrt);
    SPDLOG_WARN("mySqrt tests use {} ms",  TIMER_MSEC(mySqrt));

    SPDLOG_WARN("Running minEatingSpeed tests:");
    TIMER_START(minEatingSpeed);
    minEatingSpeed_scaffold("[3,6,7,11]", 8, 4);
    minEatingSpeed_scaffold("[30,11,23,4,20]", 5, 30);
    minEatingSpeed_scaffold("[30,11,23,4,20]", 6, 23);
    minEatingSpeed_scaffold("[10]", 3, 4);
    minEatingSpeed_scaffold("[312884470]", 312884469, 2);
    minEatingSpeed_scaffold("[332484035,524908576,855865114,632922376,222257295,690155293,"
                                "112677673,679580077,337406589,290818316,877337160,901728858,"
                                "679284947,688210097,692137887,718203285,629455728,941802184]", 823855818, 14);
    TIMER_STOP(minEatingSpeed);
    SPDLOG_WARN("minEatingSpeed tests use {} ms",  TIMER_MSEC(minEatingSpeed));

    SPDLOG_WARN("Running shipWithinDays tests:");
    TIMER_START(shipWithinDays);
    shipWithinDays_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, 15);
    shipWithinDays_scaffold("[3,2,2,4,1,4]", 3, 6);
    shipWithinDays_scaffold("[1,2,3,1,1]", 4, 3);
    shipWithinDays_scaffold("[1,2,3,4,5,6,7,8,9,10]", 10, 10);
    TIMER_STOP(shipWithinDays);
    SPDLOG_WARN("shipWithinDays tests use {} ms",  TIMER_MSEC(shipWithinDays));
}
