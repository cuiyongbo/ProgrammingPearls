#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 4, 74, 378, 668, 215 */

class Solution {
public:
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2);
    bool searchMatrix(std::vector<std::vector<int>>& matrix, int target);
    int kthSmallest(std::vector<std::vector<int>>& matrix, int k);
    int findKthNumber(int m, int n, int k);
    int findKthLargest(std::vector<int>& nums, int k);
};

int Solution::findKthLargest(std::vector<int>& nums, int k) {
/*
    Given an integer array nums and an integer k, return the kth largest element in the array.
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    Example 1:
        Input: nums = [3,2,1,5,6,4], k = 2
        Output: 5
    Example 2:
        Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
        Output: 4
*/

{ // binary search solution
    std::function<void(int, int, int)> worker = [&] (int l, int r, int d) {
        if (l >= r) {
            return;
        }
        int j = l-1;
        int pivot = nums[r];
        for (int i=l; i<r; ++i) {
            if (nums[i] > pivot) {
                std::swap(nums[++j], nums[i]);
            }
        }
        std::swap(nums[++j], nums[r]);
        int diff = j-l+1;
        //printf("l: %d, r: %d, j: %d, diff: %d, d: %d\n", l, r, j, diff, d);
        if (diff > d) {
            return worker(l, j-1, d);
        } else if (diff < d) {
            return worker(j+1, r, d-diff);
        } else {
            return;
        }
    };
    worker(0, nums.size()-1, k);
    return nums[k-1];
}

{ // std solution
    std::nth_element(nums.begin(), nums.begin()+k-1, nums.end(), std::greater<int>());
    return nums[k-1];
}

}

bool Solution::searchMatrix(std::vector<std::vector<int>>& matrix, int target) {
/*
    Write an efficient algorithm that searches for a value in an m x n matrix. 
    This matrix has the following properties:
        Integers in each row are sorted from left to right.
        The first integer of each row is greater than the last integer of the previous row.
*/
    int rows = matrix.size();
    int columns = matrix[0].size();
    int l=0;
    int r = rows*columns-1;
    while (l <= r) {
        int m = (l+r)/2;
        int a = matrix[m/columns][m%columns];
        if (a == target) {
            return true;
        } else if (a<target) {
            l = m+1;
        } else {
            r = m-1;
        }
    }
    return false;
}

double Solution::findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
/*
    There are two sorted arrays nums1 and nums2 of size m and n respectively.
    Find the median of the two sorted arrays. **The overall run time complexity should be O(log (m+n)).**

    Solution 1: first merge two arrays then find the median, however, the run time complexity would be O(m+n)
    Solution 2: use binary search to find the indices in both arrays, so that there would be ceil((n1+n2)/2) elements
    in the virtual left subarray split by the indices
*/

{ // binary search, time complexity is (log(n1)+log(n2))* log(range) ==> log(n1*n2)
    int n1 = nums1.size();
    int n2 = nums2.size();
    std::vector<int> pos;
    double s = 0;
    double ans = 0;
    if ((n1+n2)%2 == 1) {
        s = 1;
        pos.push_back((n1+n2)/2+1);
    } else {
        s = 0.5;
        pos.push_back((n1+n2)/2+1);
        pos.push_back((n1+n2)/2);
    }
    auto calc_num_less_or_eq = [] (const std::vector<int>& nums, int mid) {
        int l=0;
        int r=nums.size();
        while (l<r) {
            int m=(l+r)/2;
            if (nums[m] <= mid) {
                l = m+1;
            } else {
                r = m;
            }
        }
        return l;
    };
    while (!pos.empty()) {
        int k = pos.back(); pos.pop_back();
        int l = nums1[0] < nums2[0] ? nums1[0] : nums2[0];   
        int r = (nums1[n1-1] > nums2[n2-1] ? nums1[n1-1] : nums2[n2-1])+1;
        while (l < r) {
            int mid = (l+r)/2;
            int num = 0;
            num += calc_num_less_or_eq(nums1, mid);
            num += calc_num_less_or_eq(nums2, mid);
            if (num<k) {
                l = mid+1;
            } else {
                r = mid;
            }
        }
        ans = ans + l;
    }
    return ans*s;
}

{ // naive method, time complexity is o(m+n)
    int n1 = nums1.size();
    int n2 = nums2.size();
    std::vector<int> pos;
    double s = 0;
    if ((n1+n2)%2 == 1) {
        s = 1;
        pos.push_back((n1+n2)/2);
    } else {
        s = 0.5;
        pos.push_back((n1+n2)/2);
        pos.push_back((n1+n2)/2-1);
    }
    std::vector<std::vector<int>> mat;
    mat.push_back(nums1);
    mat.push_back(nums2);
    auto cmp = [&] (std::pair<int, int> p1, std::pair<int, int> p2) {
        return mat[p1.first][p1.second] > mat[p2.first][p2.second];
    };
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(cmp)> pq(cmp);
    pq.push(std::make_pair(0, 0));
    pq.push(std::make_pair(1, 0));
    int count = 0;
    double ans = 0;
    while (!pos.empty()) {
        auto p = pq.top(); pq.pop();
        if (count++ == pos.back()) {
            ans = ans + mat[p.first][p.second];
            pos.pop_back();
        }
        if (p.second+1 < mat[p.first].size()) {
            pq.push(std::make_pair(p.first, p.second+1));
        }
    }
    return ans*s;
}

    int n1 = nums1.size();
    int n2 = nums2.size();
    if (n1 > n2) {
        return findMedianSortedArrays(nums2, nums1);
    }

    int k = (n1+n2+1)/2; // k >= n1
    int l=0, r=n1;
    while (l < r) {
        int m = (l+r)/2; // m < n1 <= k 
        // move nums1[m] and nums2[k-m-1] close to each other
        // for boundary, use corner cases, such as n1==n2, n1=1  
        if (nums1[m] < nums2[k-m-1]) { 
            l = m+1;
        } else {
            r = m;
        }
    }

    int m1=l, m2=k-l;
    // left part size: l-1-0+1 + k-l-1-0+1 - 1 = k-1
    int c1 = std::max((m1>0) ? nums1[m1-1] : INT32_MIN,
                        (m2>0) ? nums2[m2-1] : INT32_MIN);

    // (n1+n2) must be even when c2 is needed
    // right part size: n1-l + (n2-(k-l)) - 1 = k-1
    int c2 = std::min((m1<n1) ? nums1[m1] : INT32_MAX,
                        (m2<n2) ? nums2[m2] : INT32_MAX);

    if ((n1+n2)%2 == 0) { // even count
        return (c1+c2) * 0.5;
    } else {
        return c1;
    }
}

int Solution::kthSmallest(std::vector<std::vector<int>>& matrix, int k) {
/*
    Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.
    Constraints:
        n == matrix.length == matrix[i].length
        All the rows and columns of matrix are guaranteed to be sorted in non-decreasing order.
        1 <= k <= n^2
    Hint: using a min-heap to keep the first k elements in the matrix
*/

{ // binary search solution
    int rows = matrix.size();
    int columns = matrix[0].size();
    auto calc_num_less_or_eq = [&] (int mid) {
        int ans = 0;
        for (int i=0; i<rows; ++i) {
            int l=0;
            int r=columns;
            // perform upper_bound search to figure out the number of values which are no less than mid
            while (l < r) {
                int m = (l+r)/2;
                if (matrix[i][m] <= mid) {
                    l = m+1;
                } else {
                    r = m;
                }
            }
            ans += l;
        }
        return ans;
    };
    int l = matrix[0][0];
    int r = matrix[rows-1][columns-1]+1;
    while (l < r) {
        int mid = (l+r)/2;
        int num = calc_num_less_or_eq(mid);
        if (num < k) { // lower_bound search
            l = mid+1;
        } else {
            r = mid;
        }
    }
    return l;
}

{ // min-heap solution
    typedef pair<int, int> element_type; // row, column
    auto cmp = [&] (const element_type& l, const element_type& r) {
        return matrix[l.first][l.second] > matrix[r.first][r.second];
    };
    priority_queue<element_type, std::vector<element_type>, decltype(cmp)> pq(cmp); // min-heap
    int m = matrix.size();
    for (int i=0; i<m; ++i) {
        pq.emplace(i, 0);
    }
    int ans = 0;
    for (int i=k; i!=0&&!pq.empty(); --i) {
        auto t = pq.top(); pq.pop();
        ans = matrix[t.first][t.second];
        if (t.second+1 < matrix[t.first].size()) {
            pq.emplace(t.first, t.second+1);
        }
    }
    return ans;
}

}

int Solution::findKthNumber(int m, int n, int k) {
/*
Nearly everyone has used the Multiplication Table. The multiplication table of size mxn is an integer matrix mat where mat[i][j] == i*j (1-indexed).
Given three integers m, n, and k, return the kth smallest element in the mxn multiplication table.
Examples: 
    Input: m = 3, n = 3, k = 5
    Output: 3
    Explanation: From the multiplication table below, The 5th smallest number is 3.
    1 2 3
    2 4 6
    3 6 9

// Hint: 
//  1. solution as kthSmallest; 
//  2. find the smallest element x in [1, m*n+1], such that there are k elements that are no larger than x
*/


if (0) { // min-heap solution Time Limit Exceeded
    typedef pair<int, int> element_type; // row, column
    auto cmp = [&] (const element_type& l, const element_type& r) {
        return l.first*l.second > r.first*r.second;
    };
    priority_queue<element_type, std::vector<element_type>, decltype(cmp)> pq(cmp); // min-heap
    for (int i=1; i<=m; ++i) {
        pq.emplace(i, 1);
    }
    int ans = 0;
    for (int i=k; i!=0 && !pq.empty(); --i) {
        auto t = pq.top(); pq.pop();
        ans = t.first * t.second;
        if (t.second+1 <= n) {
            pq.emplace(t.first, t.second+1);
        }
    }
    return ans;
}

{ // binary_search solution
    // since the virtual array has duplicates, we perform a lower_bound search to find the kth smallest member
    int l = 1;
    int r = m*n+1;
    while (l < r) {
        int total = 0;
        int mid = l + (r-l)/2;
        for (int i=1; i<=m && total<k; ++i) {
            total += min(mid/i, n); // number of values which are less or equal to mid
        }
        if (total < k) {
            l = mid+1;
        } else {
            r = mid;
        }
    }
    return l;
}

}

void searchMatrix_scaffold(string input, int target, bool expectedResult) {
    Solution ss;
    std::vector<std::vector<int>> matrix = stringTo2DArray<int>(input);
    bool actual = ss.searchMatrix(matrix, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findMedianSortedArrays_scaffold(string input1, string input2, double expectedResult) {
    Solution ss;
    std::vector<int> nums1 = stringTo1DArray<int>(input1);
    std::vector<int> nums2 = stringTo1DArray<int>(input2);
    double actual = ss.findMedianSortedArrays(nums1, nums2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void kthSmallest_scaffold(string input, int target, int expectedResult) {
    Solution ss;
    std::vector<std::vector<int>> matrix = stringTo2DArray<int>(input);
    int actual = ss.kthSmallest(matrix, target);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findKthNumber_scaffold(int input1, int input2, int k, int expectedResult) {
    Solution ss;
    int actual = ss.findKthNumber(input1, input2, k);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << k << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << k << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findKthLargest_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    std::vector<int> vi = stringTo1DArray<int>(input1);
    int actual = ss.findKthLargest(vi, input2);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchMatrix tests:";
    TIMER_START(searchMatrix);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 23, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 16, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 1, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 3, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 5, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 7, true);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 2, false);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 4, false);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 6, false);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 8, false);
    TIMER_STOP(searchMatrix);
    util::Log(logESSENTIAL) << "searchMatrix using " << TIMER_MSEC(searchMatrix) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findMedianSortedArrays tests:";
    TIMER_START(findMedianSortedArrays);
    findMedianSortedArrays_scaffold("[1,3]", "[2]", 2.0);
    findMedianSortedArrays_scaffold("[1,4]", "[2,3]", 2.5);
    findMedianSortedArrays_scaffold("[1,2]", "[3,4]", 2.5);
    findMedianSortedArrays_scaffold("[1,2]", "[3]", 2.0);
    TIMER_STOP(findMedianSortedArrays);
    util::Log(logESSENTIAL) << "findMedianSortedArrays using " << TIMER_MSEC(findMedianSortedArrays) << " milliseconds";

    util::Log(logESSENTIAL) << "Running kthSmallest tests:";
    TIMER_START(kthSmallest);
    kthSmallest_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 4, 7);
    kthSmallest_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 7, 16);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 8, 13);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 6, 12);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 9, 15);
    TIMER_STOP(kthSmallest);
    util::Log(logESSENTIAL) << "kthSmallest using " << TIMER_MSEC(kthSmallest) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findKthNumber tests:";
    TIMER_START(findKthNumber);
    findKthNumber_scaffold(10, 1, 5, 5);
    findKthNumber_scaffold(3, 3, 5, 3);
    findKthNumber_scaffold(3, 4, 11, 9);
    findKthNumber_scaffold(4, 3, 11, 9);
    findKthNumber_scaffold(42, 34, 401, 126);
    findKthNumber_scaffold(9895, 28405, 100787757, 31666344);
    TIMER_STOP(findKthNumber);
    util::Log(logESSENTIAL) << "findKthNumber using " << TIMER_MSEC(findKthNumber) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findKthLargest tests:";
    TIMER_START(findKthLargest);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 1, 6);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 2, 5);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 3, 4);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 4, 3);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 5, 2);
    findKthLargest_scaffold("[3,2,1,5,6,4]", 6, 1);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 1, 6);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 2, 5);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 3, 5);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 3, 4);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 4, 3);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 5, 3);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 6, 2);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 7, 2);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 8, 1);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,5,6]", 4, 5);
    TIMER_STOP(findKthLargest);
    util::Log(logESSENTIAL) << "findKthLargest using " << TIMER_MSEC(findKthLargest) << " milliseconds";
}
