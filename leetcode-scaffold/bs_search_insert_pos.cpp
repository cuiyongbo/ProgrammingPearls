#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 34, 35, 704, 981 */

class Solution {
public:
    int searchInsert(vector<int>& nums, int target);
    vector<int> searchRange(vector<int>& nums, int target);
    int binary_search(vector<int>& nums, int target);

private:
    int lower_bound(vector<int>& nums, int target);
    int upper_bound(vector<int>& nums, int target);
};


/*
    Given a sorted (in ascending order) integer array nums of n elements and a target value, 
    write a function to search target in nums. If target exists, then return its index, otherwise return -1.
*/
int Solution::binary_search(vector<int>& nums, int target) {
    int l = 0;
    int r = nums.size()-1;
    while (l<=r) {
        int m = (l+r)/2;
        if (nums[m] == target) {
            return m;
        } else if (nums[m] < target) {
            l = m+1;
        } else {
            r = m-1;
        }
    }
    return -1;
}


int Solution::lower_bound(vector<int>& nums, int target) {
    int l = 0;
    int r = nums.size();
    while (l < r) {
        int m = (l+r)/2;
        if (nums[m] < target) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


int Solution::upper_bound(vector<int>& nums, int target) {
    int l = 0;
    int r = nums.size();
    while (l < r) {
        int m = (l+r)/2;
        if (nums[m] <= target) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


/*
    Given a sorted array and a target value, return the index if the target is found. 
    If not, return the index where it would be if it were inserted in order.
    Hint: perform the lower_bound/upper_bound search.
*/
int Solution::searchInsert(vector<int>& nums, int target) {
    return lower_bound(nums, target);
}


/*
    Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
    Your algorithm’s runtime complexity must be in the order of O(log n). If the target is not found in the array, return [-1, -1].
    Hint: perform lower_bound to find the left boundray, and upper_bound for right boundary (not inclusive).
*/
vector<int> Solution::searchRange(vector<int>& nums, int target) {
    // return the first element index that is greater or equal than target
    int l = lower_bound(nums, target);
    // return the first element index that is greater than target
    int r = upper_bound(nums, target);
    if (l == r) {
        return {-1, -1};
    } else {
        return {l, r-1};
    }
}

void searchInsert_scaffold(string input, int target, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.searchInsert(nums, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, Actual: " << actual;
    }
}

void searchRange_scaffold(string input, int target, string expectedResult) {
    Solution ss;
    auto nums = stringTo1DArray<int>(input);
    auto expected = stringTo1DArray<int>(expectedResult);
    auto actual = ss.searchRange(nums, target);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
    }
}

void binary_search_scaffold(string input, int target, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.binary_search(nums, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, Actual: " << actual;
    }
}


class TimeMap {
/*
    Create a timebased key-value store class TimeMap, that supports two operations.

    1. set(string key, string value, int timestamp)
        Stores the key and value, along with the given timestamp.
    2. get(string key, int timestamp)
        Returns a value such that set(key, value, timestamp_prev) was called previously, with timestamp_prev <= timestamp.
        If there are multiple such values, it returns the one with the largest timestamp_prev. (upper_bound)
        If there are no values, it returns the empty string ("").

    Input: inputs = ["TimeMap","set","get","get","set","get","get"], 
    inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
    Output: [null,null,"bar","bar",null,"bar2","bar2"]
    Explanation:   
        TimeMap kv;   
        kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
        kv.get("foo", 1);  // output "bar"   
        kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
        kv.set("foo", "bar2", 4);   
        kv.get("foo", 4); // output "bar2"   
        kv.get("foo", 5); //output "bar2"  
*/
public:
    void set(string key, string val, int timestamp);
    string get(string key, int timestamp);

private:
    unordered_map<string, map<int, string>> m_table;
};

void TimeMap::set(string key, string val, int timestamp) {
    m_table[key][timestamp] = val;
}

string TimeMap::get(string key, int timestamp) {
    string ans;
    if (m_table.count(key) != 0) {
        const auto& it = m_table[key].upper_bound(timestamp);
        if (it != m_table[key].begin()) {
            ans = std::prev(it)->second;
        }
    }
    return ans;
}

void TimeMap_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    TimeMap tm;
    int n = (int)ans.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "set") {
            tm.set(funcArgs[i][0], funcArgs[i][1], std::stoi(funcArgs[i][2]));
        } else if (funcOperations[i] == "get") {
            string actual = tm.get(funcArgs[i][0], std::stoi(funcArgs[i][1]));
            if (actual != ans[i]) {
                util::Log(logERROR) << "get(" << funcArgs[i][0] << ", " << funcArgs[i][1] << ") failed, " 
                                    << "Expected: " << ans[i] << ", actual: " << actual;
            } else {
                util::Log(logINFO) << "get(" << funcArgs[i][0] << ", " << funcArgs[i][1] << ") passed";
            }
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchInsert tests:";
    TIMER_START(searchInsert);
    searchInsert_scaffold("[1]", 2, 1);
    searchInsert_scaffold("[1]", 0, 0);
    searchInsert_scaffold("[1,3,5,6]", 5, 2);
    searchInsert_scaffold("[1,3,5,5,5,6]", 5, 2);
    searchInsert_scaffold("[1,3,5,6]", 4, 2);
    searchInsert_scaffold("[1,3,5,6]", 7, 4);
    searchInsert_scaffold("[1,3,5,6]", 0, 0);
    TIMER_STOP(searchInsert);
    util::Log(logESSENTIAL) << "searchInsert using " << TIMER_MSEC(searchInsert) << " milliseconds";

    util::Log(logESSENTIAL) << "Running searchRange tests:";
    TIMER_START(searchRange);
    searchRange_scaffold("[1]", 2, "[-1,-1]");
    searchRange_scaffold("[1]", 0, "[-1, -1]");
    searchRange_scaffold("[1,3,5,6]", 4, "[-1, -1]");
    searchRange_scaffold("[1,3,5,6]", 5, "[2,2]");
    searchRange_scaffold("[1,3,5,5,5,6]", 5, "[2,4]");
    searchRange_scaffold("[5,7,7,8,8,10]", 8, "[3,4]");
    searchRange_scaffold("[5,7,7,8,8,10]", 6, "[-1,-1]");
    TIMER_STOP(searchRange);
    util::Log(logESSENTIAL) << "searchRange using " << TIMER_MSEC(searchRange) << " milliseconds";

    util::Log(logESSENTIAL) << "Running binary_search tests:";
    TIMER_START(binary_search);
    binary_search_scaffold("[1]", 2, -1);
    binary_search_scaffold("[1]", 0, -1);
    binary_search_scaffold("[1,3,5,6]", 5, 2);
    binary_search_scaffold("[1,3,5,6]", 0, -1);
    TIMER_STOP(binary_search);
    util::Log(logESSENTIAL) << "binary_search using " << TIMER_MSEC(binary_search) << " milliseconds";

    util::Log(logESSENTIAL) << "Running TimeMap tests:";
    TIMER_START(TimeMap);
    TimeMap_scaffold("[TimeMap,set,get,get,set,get,get]", 
                    "[[],[foo,bar,1],[foo,1],[foo,3],[foo,bar2,4],[foo,4],[foo,5]]",
                    "[null,null,bar,bar,null,bar2,bar2]");
    TimeMap_scaffold("[TimeMap,set,set,get,get,get,get,get]", 
                    "[[],[love,high,10],[love,low,20],[love,5],[love,10],[love,15],[love,20],[love,25]]",
                    "[null,null,null,,high,high,low,low]");
    TIMER_STOP(TimeMap);
    util::Log(logESSENTIAL) << "TimeMap using " << TIMER_MSEC(TimeMap) << " milliseconds";
}
