#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 62, 63, 64, 120, 174, 931, 1210 */

class Solution 
{
public:
    int uniquePaths(int m, int n);
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    int minPathSum(vector<vector<int>>& grid);
    int minimumTotal(vector<vector<int>>& t);
};

int Solution::uniquePaths(int m, int n)
{
    /*
        A robot is located at the top-left corner of a m x n grid (marked ‘Start’ in the diagram below).
        The robot can only move either down or right at any point in time. 
        The robot is trying to reach the bottom-right corner of the grid 
        (marked ‘Finish’ in the diagram below).

        How many possible unique paths are there?   
    */

    // dp[i][j] means the number of unique path to (i, j)
    // dp[i][j] = dp[i-1][j] + dp[i][j-1]

    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    dp[1][1] = 1;
    function<int(int, int)> dfs = [&](int y, int x)
    {
        if(y<0 || x<0) return 0;
        if(dp[y][x] > 0) return dp[y][x];
        int top = dfs(y-1, x);
        int left = dfs(y, x-1);
        dp[y][x] = left + top;
        return dp[y][x];
    };

    auto iterative_dfs = [&](int m, int n)
    {
        for(int r=1; r<=m; ++r)
        {
            for(int c=1; c<=n; ++c)
            {
                if(c==1 && r==1) continue;
                dp[r][c] = dp[r-1][c] + dp[r][c-1];
            }
        }
        return dp[m][n];
    };

    //return dfs(m, n);
    return iterative_dfs(m, n);
}

int Solution::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
{
    /*
        Follow up for “Unique Paths”:
        Now consider if some obstacles are added to the grids. How many unique paths would there be?
        An obstacle and empty space is marked as 1 and 0 respectively in the grid.
    */

    int m = (int)obstacleGrid.size();
    int n = (int)obstacleGrid[0].size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    dp[1][1] = (obstacleGrid[0][0] == 1) ? 0 : 1;
    auto iterative_dfs = [&](int m, int n)
    {
        for(int r=1; r<=m; ++r)
        {
            for(int c=1; c<=n; ++c)
            {
                if(c==1 && r==1) continue;
                if(obstacleGrid[r-1][c-1] == 1) continue;

                // top path
                if(r > 1 && obstacleGrid[r-2][c-1] == 0)
                    dp[r][c] += dp[r-1][c];
                
                // left path
                if(c > 1 && obstacleGrid[r-1][c-2] == 0)
                    dp[r][c] += dp[r][c-1];
            }
        }
    };

    iterative_dfs(m, n);
    return dp[m][n];
}

int Solution::minPathSum(vector<vector<int>>& grid)
{
    /*
        Given a m x n grid filled with non-negative numbers, find a path from top left 
        to bottom right which minimizes the sum of all numbers along its path.

        Note: You can only move either down or right at any point in time.
        Example 1:
            [[1,3,1],
             [1,5,1],
             [4,2,1]]
        Given the above grid map, return 7. Because the path 1→3→1→1→1 minimizes the sum.
    */

    // dp[i][j] means minPathSum to (i, j)
    // dp[i][j] = grid[i-1][j-1] + min(dp[i-1][j], dp[i][j-1]);
    int m = (int)grid.size();
    int n = (int)grid[0].size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, INT32_MAX));
    for(int i=1; i<=m; i++)
    {
        for(int j=1; j<=n; j++)
        {
            int parent = std::min(dp[i-1][j], dp[i][j-1]);
            parent = parent == INT32_MAX ? 0 : parent;
            dp[i][j] = grid[i-1][j-1] +  parent;
        }
    }
    return dp[m][n];
}

int Solution::minimumTotal(vector<vector<int>>& t)
{
    /*
        Given a triangle, find the minimum path sum from top to bottom. 
        Each step you may move to adjacent numbers on the row below.
        For example, given the following triangle:
            [
                 [2],
                [3,4],
               [6,5,7],
              [4,1,8,3]
            ]
        The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
    */

    // reuse t[i][j], t[i][j] means minCost to reach (i, j)
    // t[i][j] = t[i][j] + min(t[i-1][j], t[i-1][j-1])
    int n = (int)t.size();
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<=i; j++)
        {
            if(i==0) continue;

            if(j==0) 
                t[i][j] += t[i-1][j];
            else if(j==i)
                t[i][j] += t[i-1][j-1];
            else
                t[i][j] += std::min(t[i-1][j-1], t[i-1][j]);
        }
    }

/*
    for(const auto& v: t)
        cout << numberVectorToString(v) << endl;
*/
    return *min_element(t[n-1].begin(), t[n-1].end());
}

void uniquePaths_scaffold(int input1, int input2, int expectedResult)
{
    Solution ss;
    int actual = ss.uniquePaths(input1, input2);
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

void uniquePathsWithObstacles_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.uniquePathsWithObstacles(grid);
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

void minPathSum_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minPathSum(grid);
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

void minimumTotal_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minimumTotal(grid);
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

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running uniquePaths tests:";
    TIMER_START(uniquePaths);
    uniquePaths_scaffold(3, 7, 28);
    TIMER_STOP(uniquePaths);
    util::Log(logESSENTIAL) << "uniquePaths using " << TIMER_MSEC(uniquePaths) << " milliseconds";

    util::Log(logESSENTIAL) << "Running uniquePathsWithObstacles tests:";
    TIMER_START(uniquePathsWithObstacles);
    uniquePathsWithObstacles_scaffold("[[0,0,0],[0,1,0],[0,0,0]]", 2);
    TIMER_STOP(uniquePathsWithObstacles);
    util::Log(logESSENTIAL) << "uniquePathsWithObstacles using " << TIMER_MSEC(uniquePathsWithObstacles) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minPathSum tests:";
    TIMER_START(minPathSum);
    minPathSum_scaffold("[[1,3,1],[1,5,1],[4,2,1]]", 7);
    TIMER_STOP(minPathSum);
    util::Log(logESSENTIAL) << "minPathSum using " << TIMER_MSEC(minPathSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minimumTotal tests:";
    TIMER_START(minimumTotal);
    minimumTotal_scaffold("[[2],[3,4],[6,5,7],[4,1,8,3]]", 11);
    TIMER_STOP(minimumTotal);
    util::Log(logESSENTIAL) << "minimumTotal using " << TIMER_MSEC(minimumTotal) << " milliseconds";

}
