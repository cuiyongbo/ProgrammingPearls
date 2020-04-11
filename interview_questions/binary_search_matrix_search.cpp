#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 4, 74 */

class Solution
{
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target);
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
};

bool Solution::searchMatrix(vector<vector<int>>& matrix, int target)
{
    /*
        Write an efficient algorithm that searches for a value in an m x n matrix. 
        This matrix has the following properties:

            Integers in each row are sorted from left to right.
            The first integer of each row is greater than the last integer of the previous row.
    */

    if(matrix.empty() || matrix[0].empty())
        return false;

    int m = (int)matrix.size();
    int n = (int)matrix[0].size();

    int left = 0;
    int right = m*n - 1;
    while(left <= right)
    {
        int mid = left + (right-left)/2;
        int row = mid / n;
        int col = mid % n;

        if(matrix[row][col] == target)
        {
            return true;
        }
        else if(matrix[row][col] < target)
        {
            left = mid+1;
        }
        else
        {
            right = mid-1;
        }
    }
    return false;
}

double Solution::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
    /*
        There are two sorted arrays nums1 and nums2 of size m and n respectively.
        Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
    */

    int n1 = nums1.size();
    int n2 = nums2.size();
    if(n1 > n2)
    {
        return findMedianSortedArrays(nums2, nums1);
    }

    const int k = (n1 + n2 + 1) >> 1; // k = ceil((n1+n2)/2)

    int l = 0, r = n1;
    while(l < r)
    {
        int m1 = l + (r-l)/2;
        if(nums1[m1] < nums2[k-m1-1])
        {
            l = m1+1;
        }
        else
        {
            r = m1;
        }
    }

    int m1 = l;
    int m2 = k-l;
    int c1 = std::max(m1 <= 0 ? INT32_MIN : nums1[m1-1],
                        m2 <= 0 ? INT32_MIN : nums2[m2-1]);

    if((n1+n2)%2 == 1)
        return c1;
    
    int c2 = std::min(m1 >= n1 ? INT32_MAX : nums1[m1],
                        m2 >= n2 ? INT32_MAX : nums2[m2]);
    
    return (c1 + c2) * 0.5;
}

void searchMatrix_scaffold(string input, int target, bool expectedResult)
{
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray(input);
    bool actual = ss.searchMatrix(matrix, target);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << target << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findMedianSortedArrays_scaffold(string input1, string input2, double expectedResult)
{
    Solution ss;
    vector<int> nums1 = stringToIntegerVector(input1);
    vector<int> nums2 = stringToIntegerVector(input2);
    double actual = ss.findMedianSortedArrays(nums1, nums2);
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

    util::Log(logESSENTIAL) << "Running searchMatrix tests:";
    TIMER_START(searchMatrix);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 4, false);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 23, true);
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
}
