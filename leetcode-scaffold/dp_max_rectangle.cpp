#include "leetcode.h"

using namespace std;

/* leetcode: 85, 221, 1277 */

class Solution {
public:
    int maxRectangle(vector<vector<int>>& grid);
    int maxSquare(vector<vector<int>>& grid);
    int countSquares(vector<vector<int>>& grid);
};

int Solution::maxRectangle(vector<vector<int>>& grid) {
/*
    Given a 2D binary grid filled with 0’s and 1’s, find the largest rectangle containing only 1’s and return its area.
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
    // dp[i][j] means maxLen of all 1 sequence ending with grid[i][j] at row i
    // dp[i][j] = dp[i][j-1]+1 if grid[i][j] == 1 else 0
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns, 0));
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1) {
                dp[r][c] = (c>0) ? (dp[r][c-1]+1) : 1;
            }
        }
    }
    int ans = INT32_MIN;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            int len = INT32_MAX;
            for (int k=r; k<rows; ++k) {
                // maxRectangle in grid[r:, c]
                len = min(len, dp[k][c]);
                if (len == 0) { // stop when encountering a zero-row
                    break;
                }
                ans = max(ans, len*(k-r+1));
            }
        }
    }
    return ans;
}

int Solution::maxSquare(vector<vector<int>>& grid) {
/*
    Given a 2D binary grid filled with 0’s and 1’s, find the largest square containing only 1’s and return its area.
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
    int rows = grid.size();
    int columns = grid[0].size();
    // dp[i][j] emans maximum length of all 1 sequence ending with grid[i][j] at row i
    // dp[i][j] = dp[i][j-1]+1 if grid[i][j]==1 else 0
    vector<vector<int>> dp(rows, vector<int>(columns, 0));
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1) {
                dp[r][c] = (c>0) ? (dp[r][c-1]+1) : 1;
            } else {
                dp[r][c] = 0;
            }
        }
    }
    int ans = 0;
    for (int r=0; r<rows; r++) {
        for (int c=0; c<columns; c++) {
            // maxSquare for grid[r:, c]
            int len = INT32_MAX;
            for (int k=r; k<rows; k++) {
                len = min(len, dp[k][c]);
                if (len == 0) { // stop at row with all 0
                    break;
                }
                int w = min(len, k-r+1);
                ans = max(ans, w*w);
            }
        }
    }
    return ans;
}


int Solution::countSquares(vector<vector<int>>& grid) {
/*
    Given a m * n grid of 1 and 0, return how many square submatrices have all 1.
    Example:
        Input: grid =
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
    // dp[i][j] means side of the largest square with bottom-right corner located at (i, j)
    // dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 if grid[i][j]==1 else 0
    // ans = sum(dp)
    int ans = 0;
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<int>> dp(rows+1, vector<int>(columns+1, 0));
    for (int r=1; r<=rows; ++r) {
        for (int c=1; c<=columns; ++c) {
            if (grid[r-1][c-1] == 1) {
                dp[r][c] = min({dp[r-1][c], dp[r][c-1], dp[r-1][c-1]}) + 1;
                ans += dp[r][c];
            }
        }
    }
    return ans;
}


void maxRectangle_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.maxRectangle(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void maxSquare_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.maxSquare(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void countSquares_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.countSquares(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_INFO("Running maxRectangle tests:");
    TIMER_START(maxRectangle);
    maxRectangle_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 4);
    maxRectangle_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    maxRectangle_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 6);
    TIMER_STOP(maxRectangle);
    SPDLOG_INFO("maxRectangle tests use {} ms", TIMER_MSEC(maxRectangle));

    SPDLOG_INFO("Running maxSquare tests:");
    TIMER_START(maxSquare);
    maxSquare_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 4);
    maxSquare_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    maxSquare_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 4);
    TIMER_STOP(maxSquare);
    SPDLOG_INFO("maxSquare tests use {} ms", TIMER_MSEC(maxSquare));

    SPDLOG_INFO("Running countSquares tests:");
    TIMER_START(countSquares);
    countSquares_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 13);
    countSquares_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 27);
    countSquares_scaffold("[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]", 15);
    countSquares_scaffold("[[0,1,1,1],[1,1,1,1],[0,1,1,1]]", 15);
    countSquares_scaffold("[[1,0,1],[1,1,0],[1,1,0]]", 7);
    TIMER_STOP(countSquares);
    SPDLOG_INFO("countSquares tests use {} ms", TIMER_MSEC(countSquares));
}
