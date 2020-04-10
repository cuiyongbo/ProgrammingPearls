#include "leetcode.h"
#include "util/trip_farthest_insertion.hpp"

using namespace std;
using namespace osrm;

/* leetcode exercises: 74 */

class Solution
{
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target);
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

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchMatrix tests:";
    TIMER_START(searchMatrix);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 4, false);
    searchMatrix_scaffold("[[1,3,5,7], [10, 11, 16, 20], [23, 30, 34, 50]]", 23, true);
    TIMER_STOP(searchMatrix);
    util::Log(logESSENTIAL) << "searchMatrix using " << TIMER_MSEC(searchMatrix) << " milliseconds";
}
