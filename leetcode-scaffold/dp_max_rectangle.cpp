#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 85, 221, 304, 1277 */

class Solution 
{
public:
    int maxRectangle(vector<vector<int>>& grid);
    int maxSquare(vector<vector<int>>& matrix);
    int countSquares(vector<vector<int>>& matrix);
};

int Solution::maxRectangle(vector<vector<int>>& grid)
{
    /*
        Given a 2D binary matrix filled with 0’s and 1’s, 
        find the largest rectangle containing only 1’s and return its area.

        Example:
            Input:
            [
              ["1","0","1","0","0"],
              ["1","0","1","1","1"],
              ["1","1","1","1","1"],
              ["1","0","0","1","0"]
            ]
        Output: 6
    */

    // dp[i][j] := max length of all 1 sequence ends with col j, at the i-th row.
    // transition:
    //    dp[i][j] = 0 if matrix[i][j] == ‘0’
    //             = dp[i][j-1] + 1 if matrix[i][j] == ‘1’

    int m  = (int)grid.size();
    int n  = (int)grid[0].size();
    vector<vector<int>> dp(m, vector<int>(n));
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            dp[i][j] = (grid[i][j] == 1) ? (j==0 ? 1 : dp[i][j-1]+1) : 0;
    }

    int ans = 0;
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            // maxRectangle of grid[i:k, :j]
            int len = INT32_MAX;
            for(int k=i; k<m; k++)
            {
                len = std::min(len, dp[k][j]);
                if(len == 0) break;
                ans = std::max(ans, len*(k-i+1));
            }
        }
    }
    return ans;
}

int Solution::maxSquare(vector<vector<int>>& grid)
{
    /*
        Given a 2D binary matrix filled with 0’s and 1’s, 
        find the largest square containing only 1’s and return its area.

        Example:
            Input:
            [
              ["1","0","1","0","0"],
              ["1","0","1","1","1"],
              ["1","1","1","1","1"],
              ["1","0","0","1","0"]
            ]
        Output: 4
    */

    // dp[i][j] := max length of all 1 sequence ends with col j, at the i-th row.
    // transition:
    //    dp[i][j] = 0 if matrix[i][j] == ‘0’
    //             = dp[i][j-1] + 1 if matrix[i][j] == ‘1’

    int m  = (int)grid.size();
    int n  = (int)grid[0].size();
    vector<vector<int>> dp(m, vector<int>(n));
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
            dp[i][j] = (grid[i][j] == 0) ? 0 : (j==0 ? 1 : dp[i][j-1]+1);
    }

    int ans = 0;
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            // maxSquare of grid[i:k, :j]
            int len = INT32_MAX;
            for(int k=i; k<m; k++)
            {
                len = std::min(len, dp[k][j]);
                if(len < k-i+1) break;
                ans = std::max(ans, k-i+1);
            }
        }
    }
    return ans*ans;
}

int Solution::countSquares(vector<vector<int>>& matrix)
{
    /*
        Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.

        Example 1:
        Input: matrix =
        [
          [0,1,1,1],
          [1,1,1,1],
          [0,1,1,1]
        ]
        Output: 15
        Explanation: 
        There are 10 squares of side 1.
        There are 4 squares of side 2.
        There is  1 square of side 3.
        Total number of squares = 10 + 4 + 1 = 15.
    */

    // dp[i][j] := edge of largest square with bottom right corner at (i, j)
    // dp[i][j] = min(dp[i–1][j], dp[i–1][j–1], dp[i][j–1]) if matrix[i][j] == 1 else 0
    // ans = sum(dp)

    int ans = 0;
    int m = (int)matrix.size();
    int n = (int)matrix[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(matrix[i][j] == 0) continue;

            dp[i][j] = 1;
            if(i>0 && j>0) dp[i][j] = std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1; 
            ans += dp[i][j];
        }
    }
    return ans;
}

void maxRectangle_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.maxRectangle(grid);
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

void maxSquare_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.maxSquare(grid);
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

void countSquares_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.countSquares(grid);
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

/*
    Given a 2D matrix matrix, find the sum of the elements inside the rectangle 
    defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
    (row1, col1), (row2, col2) are 0-indexed.
*/

class NumMatrix
{
public:
    NumMatrix(const vector<vector<int>>& matrix);
    int sumRegion(int row1, int col1, int row2, int col2);
private:
    vector<vector<int>> m_dp;
};

NumMatrix::NumMatrix(const vector<vector<int>>& matrix): m_dp(matrix)
{
    // dp[i][j] = sumRegion(0, 0, i, j)
    int m = (int)matrix.size();
    int n = (int)matrix[0].size();
    for(int i=0; i<m; i++)
    {
        for(int j=1; j<n; j++)
        {
            m_dp[i][j] += m_dp[i][j-1];
        }
    }

    for(int i=1; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            m_dp[i][j] += m_dp[i-1][j];
        }
    }
}

int NumMatrix::sumRegion(int row1, int col1, int row2, int col2)
{
    int ans = m_dp[row2][col2];
    if(row1 > 0) ans -= m_dp[row1-1][col2];
    if(col1 > 0) ans -= m_dp[row2][col1-1];
    if(row1 > 0 && col1 > 0) ans += m_dp[row1-1][col1-1];
    return ans;
}

void NumMatrix_scaffold(string input, string operations, string args, string expectedOutputs)
{
    vector<vector<int>> matrix = stringTo2DArray<int>(input);
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<int>> funcArgs = stringTo2DArray<int>(args);
    vector<int> ans = stringTo1DArray<int>(expectedOutputs);
    NumMatrix tm(matrix);
    int n = (int)ans.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "sumRegion")
        {
            int actual = tm.sumRegion(funcArgs[i][0], funcArgs[i][1], funcArgs[i][2], funcArgs[i][3]);
            if(actual != ans[i])
            {
                util::Log(logERROR) << funcOperations[i] << "(" << numberVectorToString<int>(funcArgs[i]) << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << funcOperations[i] << "(" << numberVectorToString<int>(funcArgs[i]) << ") passed";
            }
        }
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running maxRectangle tests:";
    TIMER_START(maxRectangle);
    maxRectangle_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 4);
    maxRectangle_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    maxRectangle_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 6);
    TIMER_STOP(maxRectangle);
    util::Log(logESSENTIAL) << "maxRectangle using " << TIMER_MSEC(maxRectangle) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxSquare tests:";
    TIMER_START(maxSquare);
    maxSquare_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 4);
    maxSquare_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    maxSquare_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 4);
    TIMER_STOP(maxSquare);
    util::Log(logESSENTIAL) << "maxSquare using " << TIMER_MSEC(maxSquare) << " milliseconds";

    util::Log(logESSENTIAL) << "Running countSquares tests:";
    TIMER_START(countSquares);
    countSquares_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 13);
    countSquares_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 27);
    countSquares_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 15);
    countSquares_scaffold("[[0,1,1,1],[1,1,1,1],[0,1,1,1]]", 15);
    countSquares_scaffold("[[1,0,1],[1,1,0],[1,1,0]]", 7);
    TIMER_STOP(countSquares);
    util::Log(logESSENTIAL) << "countSquares using " << TIMER_MSEC(countSquares) << " milliseconds";

    string input1 = R"([
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]])";

    string operations = "[sumRegion,sumRegion,sumRegion]";
    string args = "[[2, 1, 4, 3],[1, 1, 2, 2],[1, 2, 2, 4]]";
    string expectedOutputs = "[8,11,12]";

    util::Log(logESSENTIAL) << "Running NumMatrix tests:";
    TIMER_START(NumMatrix);
    NumMatrix_scaffold(input1, operations, args, expectedOutputs);
    TIMER_STOP(NumMatrix);
    util::Log(logESSENTIAL) << "NumMatrix using " << TIMER_MSEC(NumMatrix) << " milliseconds";
}
