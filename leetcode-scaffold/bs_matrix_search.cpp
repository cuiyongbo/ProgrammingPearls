#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 4, 74, 378, 668, 215 */

class Solution {
public:
    int findKthLargest(std::vector<int>& nums, int k);
    int kthSmallest(std::vector<std::vector<int>>& matrix, int k);
    int findKthNumber(int m, int n, int k);
    bool searchMatrix(std::vector<std::vector<int>>& matrix, int target);
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2);
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
    Hint: you may use quicksort like algorithm to sort the array partially, then fetch the kth element
*/

if (0) { // std solution
    std::nth_element(nums.begin(), nums.begin()+k-1, nums.end(), std::greater<int>());
    return nums[k-1];
}

{ // iterative version
    auto partitioner = [&] (int l, int r) {
        int pivot = nums[r];
        int j = l-1;
        for (int i=l; i<r; i++) {
            if (nums[i] > pivot) {
                swap(nums[i], nums[++j]);
            }
        }
        swap(nums[++j], nums[r]);
        return j;
    };
    int kth = k; // save k for later use
    int l = 0;
    int r = nums.size()-1; // r is inclusive
    while (l<=r) {
        int m = partitioner(l, r);
        int count = m-l+1;
        if (count == k) {
            break;
        } else if (count < k) { // target resides in right part
            l = m+1;
            k = k - count; // skip elements which are larger than target but may not necessarily be sorted
        } else { // target resides in left part
            r = m-1;
            //k = k;
        }
    }
    return nums[kth-1];
}

{ // binary search solution, recursive version
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
    int l = 0;
    int r = rows*columns-1; // r is inclusive
    while (l <= r) {
        int m = (l+r)/2;
        // convert m to (row,cloumn)
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

{ // naive solution
    vector<int> target_positions;
    int total_count = nums1.size() + nums2.size();
    if (total_count % 2 == 0) {
        int k = total_count/2;
        target_positions.push_back(k);
        target_positions.push_back(k+1);

    } else {
        int k = (total_count+1)/2;
        target_positions.push_back(k);
    }
    // perform upper_bound search to find the number of elements which are no larger than target
    auto worker = [](const std::vector<int>& nums, int target) {
        auto it = std::upper_bound(nums.begin(), nums.end(), target);
        return std::distance(nums.begin(), it);
    };
    // perform lower_bound search to find the k-th element of the two arrays
    auto find_kth_element = [&] (int k) {
        int l = min(nums1.front(), nums2.front());
        int r = max(nums1.back(), nums2.back()) + 1; // r is not inclusive
        while (l<r) {
            int m = (l+r)/2;
            int count = 0;
            count += worker(nums1, m);
            count += worker(nums2, m);
            if (count < k) {
                l = m+1;
            } else {
                r = m;
            }
        }
        return l;
    };
    double sum = 0;
    for (auto k: target_positions) {
        sum += find_kth_element(k);
    }
    return sum/target_positions.size();
}


{
    // for case: nums1=[2], nums2=[]
    if (nums1.size() > nums2.size()) {
        return findMedianSortedArrays(nums2, nums1);
    }

    int x = nums1.size();
    int y = nums2.size();
    int l = 0;
    int r = x;
    while (l <= r) {
        int partitionX = (l+r)/2;
        int partitionY = (x+y+1)/2 - partitionX;
        int maxLeftX = (partitionX == 0) ? INT32_MIN : nums1[partitionX-1];
        int minRightX = (partitionX == x) ? INT32_MAX : nums1[partitionX];
        int maxLeftY = (partitionY == 0) ? INT32_MIN : nums2[partitionY-1];
        int minRightY = (partitionY == y) ? INT32_MAX : nums2[partitionY];
        if ((maxLeftX<=minRightY )&& (maxLeftY<=minRightX)) {
            // corrected partition found
            if ((x+y)%2 == 0) {
                return (std::max(maxLeftX, maxLeftY) + std::min(minRightX, minRightY)) * 0.5;
            } else {
                return std::max(maxLeftX, maxLeftY);
            }
        } else if (maxLeftX > minRightY) {
            // we are too far on the right side of partitionX, so go left
            r = partitionX-1;
        } else {
            // we are too far on the left side of partitionX, so go right
            l = partitionX+1;
        }
    }
    return 0;
}

}

int Solution::kthSmallest(std::vector<std::vector<int>>& matrix, int k) {
/*
    Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.
    Constraints:
        n == matrix.length == matrix[i].length
        All the rows and columns of matrix are guaranteed to be sorted in non-decreasing order.
        1 <= k <= n^2
    Hint: using a min-heap/max-heap to keep the first k elements in the matrix
*/

{
    int rows = matrix.size();
    int columns = matrix[0].size();
    auto calc_num_no_larger = [&] (int target) {
        int ans = 0;
        for (int i=0; i<rows; i++) {
            auto it = std::upper_bound(matrix[i].begin(), matrix[i].end(), target);
            ans += std::distance(matrix[i].begin(), it);
        }
        return ans;
    };
    // perform lower_bound search to find the answer
    int l = 0;
    int r = matrix[rows-1][columns-1] + 1;
    while (l<r) {
        int m = (l+r)/2;
        // perform upper_bound search to find the number of elements that are no larger than target in matrix
        int num = calc_num_no_larger(m);
        if (num < k) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


{ // binary search solution
    int rows = matrix.size();
    int columns = matrix[0].size();
    auto calc_num_less_or_eq = [&] (int mid) {
        int ans = 0;
        for (int i=0; i<rows; ++i) {
            int l=0;
            int r=columns;
            // perform `upper_bound` search to find the number of values which are greater than `mid`
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

{ // max-heap
    int m = matrix.size();
    int n = matrix[0].size();
    std::priority_queue<int, vector<int>, std::less<int>> pq;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (pq.size() < k) {
                pq.push(matrix[i][j]);
            } else {
                if (matrix[i][j] < pq.top()) {
                    pq.pop(); pq.push(matrix[i][j]);
                }
            }
        }
    }
    return pq.top();
}

}

int Solution::findKthNumber(int rows, int columns, int k) {
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

Hint: 
 1. solution as kthSmallest; 
 2. find the smallest element x in [1, m*n+1], such that there are k elements that are no larger than x
*/
    int l = 1;
    int r = rows*columns+1; // r is not inclusive
    while (l<r) {
        int total_count = 0;
        int m = (l+r)/2;
        for (int i=1; i<=rows; i++) {
            total_count += std::min(m/i, columns);
        }
        if (total_count < k) {
            l = m+1;
        } else {
            r = m;
        }
    }
    return l;
}


void searchMatrix_scaffold(string input, int target, bool expectedResult) {
    Solution ss;
    std::vector<std::vector<int>> matrix = stringTo2DArray<int>(input);
    bool actual = ss.searchMatrix(matrix, target);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, target, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, target, expectedResult, actual);
    }
}


void findMedianSortedArrays_scaffold(string input1, string input2, double expectedResult) {
    Solution ss;
    std::vector<int> nums1 = stringTo1DArray<int>(input1);
    std::vector<int> nums2 = stringTo1DArray<int>(input2);
    double actual = ss.findMedianSortedArrays(nums1, nums2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


void kthSmallest_scaffold(string input, int target, int expectedResult) {
    Solution ss;
    std::vector<std::vector<int>> matrix = stringTo2DArray<int>(input);
    int actual = ss.kthSmallest(matrix, target);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, target, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, target, expectedResult, actual);
    }
}


void findKthNumber_scaffold(int input1, int input2, int k, int expectedResult) {
    Solution ss;
    int actual = ss.findKthNumber(input1, input2, k);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, {}, expectedResult={}) passed", input1, input1, k, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, expectedResult={}) failed, actual: {}", input1, input1, k, expectedResult, actual);
    }
}


void findKthLargest_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    std::vector<int> vi = stringTo1DArray<int>(input1);
    int actual = ss.findKthLargest(vi, input2);
    if(actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    SPDLOG_WARN("Running searchMatrix tests:");
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
    SPDLOG_WARN("searchMatrix tests use {} ms", TIMER_MSEC(searchMatrix));

    SPDLOG_WARN("Running findMedianSortedArrays tests:");
    TIMER_START(findMedianSortedArrays);
    findMedianSortedArrays_scaffold("[1,3]", "[2]", 2.0);
    findMedianSortedArrays_scaffold("[1,4]", "[2,3]", 2.5);
    findMedianSortedArrays_scaffold("[1,2]", "[3,4]", 2.5);
    findMedianSortedArrays_scaffold("[1,2]", "[3]", 2.0);
    TIMER_STOP(findMedianSortedArrays);
    SPDLOG_WARN("findMedianSortedArrays tests use {} ms", TIMER_MSEC(findMedianSortedArrays));

    SPDLOG_WARN("Running kthSmallest tests:");
    TIMER_START(kthSmallest);
    kthSmallest_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 4, 7);
    kthSmallest_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 7, 16);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 8, 13);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 6, 12);
    kthSmallest_scaffold("[[1,5,9], [10,11,13], [12,13,15]]", 9, 15);
    TIMER_STOP(kthSmallest);
    SPDLOG_WARN("kthSmallest tests use {} ms", TIMER_MSEC(kthSmallest));

    SPDLOG_WARN("Running findKthNumber tests:");
    TIMER_START(findKthNumber);
    findKthNumber_scaffold(10, 1, 5, 5);
    findKthNumber_scaffold(3, 3, 5, 3);
    findKthNumber_scaffold(3, 4, 11, 9);
    findKthNumber_scaffold(4, 3, 11, 9);
    findKthNumber_scaffold(42, 34, 401, 126);
    findKthNumber_scaffold(9895, 28405, 100787757, 31666344);
    TIMER_STOP(findKthNumber);
    SPDLOG_WARN("findKthNumber tests use {} ms", TIMER_MSEC(findKthNumber));

    SPDLOG_WARN("Running findKthLargest tests:");
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
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 4, 4);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 5, 3);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 6, 3);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 7, 2);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 8, 2);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,6]", 9, 1);
    findKthLargest_scaffold("[3,2,3,1,2,4,5,5,5,6]", 4, 5);
    TIMER_STOP(findKthLargest);
    SPDLOG_WARN("findKthLargest tests use {} ms", TIMER_MSEC(findKthLargest));
}
