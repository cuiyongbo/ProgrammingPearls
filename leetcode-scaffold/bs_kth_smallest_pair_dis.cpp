#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 719, 786 */

class Solution {
public:
    int smallestDistancePair(std::vector<int>& nums, int k);
    std::vector<int> kthSmallestPrimeFraction(std::vector<int>& A, int K);
};


int Solution::smallestDistancePair(std::vector<int>& nums, int k) {
/*
    Given an integer array, return the k-th smallest distance among all the pairs. The distance of a pair (A, B) is defined as the absolute difference between A and B.
    Constraints:
        n == nums.length
        1 <= k <= n * (n - 1) / 2
*/

if (0) { // naive solution, O(n^2)
    map<int, int> mp; // distance, pair_num
    int sz = nums.size();
    for (int i=0; i<sz; ++i) {
        for (int j=i+1; j<sz; ++j) {
            mp[abs(nums[i]-nums[j])]++;
        }
    }
    int ans = 0;
    int count = 0;
    // traverse the map by key in ascending order
    for (auto it: mp) {
        count += it.second;
        if (count >= k) { // lower bound
            ans = it.first;
            break;
        }
    }
    return ans;
}

{ // binary search, O(log(max(nums)-min(nums))*log(n)) + O(nlogn) -> O(nlogn)
    // sorted the array in ascending order so than we can perform upper_bound search with it
    std::sort(nums.begin(), nums.end());
    int sz = nums.size();
    auto calc_num_pairs = [&] (int distance) {
        int ans = 0;
        for (int i=0; i<sz; ++i) {
            int l=i+1, r=sz; // r is not inclusive
            while (l < r) {
                int m = (l+r)/2;
                if (nums[m]-nums[i] <= distance) { // when diff==key we move l to right to find the leftmost element that is greater than key
                    l = m+1;
                } else {
                    r = m;
                }
            }
            ans += l-i-1;
        }
        return ans;
    };
    int l = 0;
    int r = nums[sz-1]-nums[0]+1; // r is not inclusive
    while (l < r) {
        int m = (r+l)/2;
        // perform upper_bound search to figure out the number of pairs whose distance is no larger than m
        int pair_num = calc_num_pairs(m);
        // perform lower_bound search to find the first m such that total is no less than k
        if (pair_num < k) {
            l = m+1;
        } else { // when diff==key we move r to left to find the leftmost element that is greater than or equal to key
            r = m;
        }
    }
    return l;
}

}


std::vector<int> Solution::kthSmallestPrimeFraction(std::vector<int>& nums, int k) {
/*
    A **sorted** list A contains 1, plus some number of primes. Then for every p < q in the list, we consider the fraction p/q.
    What is the K-th smallest fraction considered?  Return your answer as an array of ints, where answer[0] = p and answer[1] = q. 

    Examples:
        Input: A = [1, 2, 3, 5], K = 3
        Output: [2, 5]
        Explanation: The fractions to be considered in sorted order are: 1/5, 1/3, 2/5, 1/2, 3/5, 2/3. The third fraction is 2/5.
*/

{ // naive version
    int sz = nums.size();
    std::vector<int> ret{nums[0], nums[1]};
    auto calc_num_less = [&] (double m) {
        int ans = 0;
        double max_f = 0;
        for (int i=0; i<sz; ++i) {
            int l = i+1;
            int r = sz;
            while (l < r) {
                int j = (l+r)/2;
                if (nums[i] >= m*nums[j]) {
                    l=j+1;
                } else {
                    r=j;
                }
            }
            //printf("i: %d, l: %d, sz: %d, m: %f\n", i, l, sz, m);
            if (l != sz && nums[i] > max_f*nums[l]) {
                max_f = double(nums[i])/nums[l];
                ret[0] = nums[i]; ret[1] = nums[l];
            }
            ans += (sz-l);
        }
        return ans;
    };

    double l = 0;
    double r = 1.0;
    while (l < r) {
        double m = (l+r)/2;
        int num_less = calc_num_less(m);
        if (num_less < k) {
            l=m;
        } else if (num_less > k) {
            r=m;
        } else {
            break;
        }
    }
    return ret;
}


{ // refined version  
    int sz = nums.size();
    double l = 0.0;
    double r = 1.0;
    while (l<r) {
        double m = (l+r)/2;
        int total = 0;
        int p=0, q=0; double max_f = 0;
        // find the number of pairs whose fraction is no greater than m
        for (int i=0, j=1; i<sz-1; ++i) {
            // j = i+1; // no need, if we have nums[i]/nums[j] > m, then nums[i+1]/nums[j] > m since nums is sorted in ascending order
            while (j<sz && nums[i] > m*nums[j]) {
            //while (j<sz && nums[i]/nums[j] > m) {
                ++j;
            }
            if (sz == j) { // no need go further, since nums[i]/nums[j-1] is the smallest
                break;
            }
            total += (sz-j);
            if (nums[i] > max_f * nums[j]) {
                max_f = nums[i]/nums[j];
                p = nums[i]; q = nums[j];
            }
        }
        if (total == k) {
            return {p, q};
        } else if (total < k) {
            l = m;
        } else {
            r = m;
        }
    } 
    return {};
}

}


void smallestDistancePair_scaffold(string input, int target, int expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.smallestDistancePair(nums, target);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, target, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, target, expectedResult, actual);
    }
}


void kthSmallestPrimeFraction_scaffold(string input, int target, string expectedResult) {
    Solution ss;
    std::vector<int> nums = stringTo1DArray<int>(input);
    std::vector<int> expected = stringTo1DArray<int>(expectedResult);
    std::vector<int> actual = ss.kthSmallestPrimeFraction(nums, target);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, target, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, target, expectedResult, numberVectorToString(actual));
    }
}


int main() {
    SPDLOG_WARN("Running smallestDistancePair tests:");
    TIMER_START(smallestDistancePair);
    smallestDistancePair_scaffold("[1,3,1]", 1, 0);
    smallestDistancePair_scaffold("[1,1,1]", 2, 0);
    smallestDistancePair_scaffold("[1,6,1]", 3, 5);
    smallestDistancePair_scaffold("[62,100,4]", 2, 58);
    TIMER_STOP(smallestDistancePair);
    SPDLOG_WARN("smallestDistancePair tests use {} ms", TIMER_MSEC(smallestDistancePair));

    SPDLOG_WARN("Running kthSmallestPrimeFraction tests:");
    TIMER_START(kthSmallestPrimeFraction);
    kthSmallestPrimeFraction_scaffold("[1,7]", 1, "[1,7]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 1, "[1,5]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 2, "[1,3]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 3, "[2,5]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 4, "[1,2]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 5, "[3,5]");
    kthSmallestPrimeFraction_scaffold("[1,2,3,5]", 6, "[2,3]");
    TIMER_STOP(kthSmallestPrimeFraction);
    SPDLOG_WARN("kthSmallestPrimeFraction tests use {} ms", TIMER_MSEC(kthSmallestPrimeFraction));
}
