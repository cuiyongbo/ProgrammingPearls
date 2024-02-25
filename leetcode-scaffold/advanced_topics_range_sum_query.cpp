#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 307 */

/*
    Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
    The update(i, val) function modifies nums by updating the element at index i to val.
    Example:
        Given nums = [1, 3, 5]
        sumRange(0, 2) -> 9
        update(1, 2)
        sumRange(0, 2) -> 8
    Note:
    
    The array is only modifiable by the update function.
    You may assume the number of calls to update and sumRange function is distributed evenly.
*/

class FenwickTree {    
public:
    FenwickTree(const vector<int>& nums);

    // 1-indexed array
    void update(int i, int delta);
    int query(int i) const;

private:
    // get the least significant one
    int lsb(int x) const { return x & (-x); }

private:
    // m_ft[i] is responsible for elments in range [i-lsb(i)+1, i]
    vector<int> m_ft;
};

FenwickTree::FenwickTree(const vector<int>& nums) {
    int n = nums.size();
    m_ft.assign(n+1, 0);
    for(int i=0; i<n; i++) {
        update(i+1, nums[i]);
    }
}

void FenwickTree::update(int i, int delta) {
    // Adding lsb(x) to x allows it to traverse its responsibility tree upwards.
    while (i < m_ft.size()) {
        m_ft[i] += delta;
        i += lsb(i);
    }
}

int FenwickTree::query(int i) const {
    // Subtracting lsb(x) from x gives the largest index that is out of responsibility of x.
    int sum = 0;
    while (i>0) {
        sum += m_ft[i];
        i -= lsb(i);
    }
    return sum;
}
 
class NumArray {
public:
    NumArray(const vector<int>& nums)
    : m_sum(nums), m_ft(nums) {
    }

    void update(int i, int val);
    int sumRange(int i, int j);

private:
    vector<int> m_sum;
    FenwickTree m_ft;
};

void NumArray::update(int i, int val) {
    m_ft.update(i+1, val-m_sum[i]);
    m_sum[i] = val;
}

int NumArray::sumRange(int i, int j) {
    return m_ft.query(j+1) - m_ft.query(i);
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running NumArray tests:";
    TIMER_START(NumArray);
    NumArray na({1, 3, 5});
    assert(na.sumRange(0,2) == 9);
    na.update(1, 2);
    assert(na.sumRange(0,2) == 8);
    TIMER_STOP(NumArray);
    util::Log(logESSENTIAL) << "NumArray using " << TIMER_MSEC(NumArray) << " milliseconds";
}
