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
    int calculateMinimumHP(vector<vector<int>>& dungeon);
    int minFallingSum(vector<vector<int>>& grid);
    int minimumMoves(vector<vector<int>>& grid);
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

int Solution::minFallingSum(vector<vector<int>>& grid)
{
    /*
        Given a square array of integers A, we want the minimum sum of a falling path through A.
        A falling path starts at any element in the first row, and chooses one element from each row.  
        The next row’s choice must be in a column that is different from the previous row’s column by at most one.

        Example 1:
            Input: [[1,2,3],[4,5,6],[7,8,9]]
            Output: 12
            The possible falling paths are:
                [1,4,7], [1,4,8], [1,5,7], [1,5,8], [1,5,9]
                [2,4,7], [2,4,8], [2,5,7], [2,5,8], [2,5,9], [2,6,8], [2,6,9]
                [3,5,7], [3,5,8], [3,5,9], [3,6,8], [3,6,9]
            The falling path with the smallest sum is [1,4,7], so the answer is 12.
    */

    int m = (int)grid.size();
    int n = (int)grid[0].size();
    for(int i=1; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(j==0 && j==n-1)
                grid[i][j] += grid[i-1][j];
            else if(j == 0) 
                grid[i][j] += std::min(grid[i-1][j], grid[i-1][j+1]);
            else if(j == n-1)
                grid[i][j] += std::min(grid[i-1][j], grid[i-1][j-1]);
            else
                grid[i][j] += std::min(std::min(grid[i-1][j], grid[i-1][j-1]), grid[i-1][j+1]);
        }
    }
    return *min_element(grid[m-1].begin(), grid[m-1].end());
}

int Solution::minimumMoves(vector<vector<int>>& grid)
{
    /*
        In an n*n grid, there is a snake that spans 2 cells and starts moving from the top left 
        corner at (0, 0) and (0, 1). The grid has empty cells represented by 0 and blocked 
        cells represented by 1. The snake wants to reach the lower right corner at (n-1, n-2) 
        and (n-1, n-1).

        In one move the snake can:

            Move one cell to the right if there are no blocked cells there. 
            This move keeps the horizontal/vertical position of the snake as it is.

            Move down one cell if there are no blocked cells there. 
            This move keeps the horizontal/vertical position of the snake as it is.
            
            Rotate clockwise if it’s in a horizontal position and the two cells 
            under it are both empty. In that case the snake moves from (r, c) 
            and (r, c+1) to (r, c) and (r+1, c).
            
            Rotate counterclockwise if it’s in a vertical position and the two cells 
            to its right are both empty. In that case the snake moves from (r, c) 
            and (r+1, c) to (r, c) and (r, c+1).
            
        Return the minimum number of moves to reach the target.
        If there is no way to reach the target, return -1.

        Example:

        Input: 
        grid = [[0,0,0,0,0,1],
                [1,1,0,0,1,0],
                [0,0,0,0,1,1],
                [0,0,1,0,1,0],
                [0,1,1,0,0,0],
                [0,1,1,0,0,0]]
        Output: 11
        Explanation:
        One possible solution is [right, right, rotate clockwise, right, down, 
        down, down, down, rotate counterclockwise, right, down].

        Example 2:
        Input: grid = [[0,0,1,1,1,1],
                       [0,0,0,0,1,1],
                       [1,1,0,0,0,1],
                       [1,1,1,0,0,1],
                       [1,1,1,0,0,1],
                       [1,1,1,0,0,0]]
        Output: 9

        Constraints:

            2 <= n <= 100
            0 <= grid[i][j] <= 1
            It is guaranteed that the snake starts at empty cells.
    */

    // used[i][j] = 0: not used
    // used[i][j] = 1: tail facing right used
    // used[i][j] = 2: tail facing down used
    // used[i][j] = 3: done

    int n  = (int)grid.size();
    vector<vector<int>> used(n, vector<int>(n, 0));
    auto is_right_used = [&](int y, int x) { return (used[y][x] & 0x1) != 0;};
    auto is_down_used = [&](int y, int x) { return (used[y][x] & 0x2) != 0;};
    auto set_right = [&](int y, int x) { used[y][x] |= 0x1;};
    auto set_down = [&](int y, int x) { used[y][x] |= 0x2;};

    auto is_target = [&] (const vector<int>& t)
    {
        return t[0] == n-1 && t[1] == n-2 && t[2] == n-1 && t[3] == n-1;
    };

    int steps = 0;
    set_right(0, 1);
    queue<vector<int>> q;
    q.push({0, 0, 0, 1}); // y1, x1, y2, x2
    while(!q.empty())
    {
        for(size_t i=q.size(); i != 0; i--)
        {
            auto t = q.front(); q.pop();
            if(is_target(t)) return steps;

            int y = t[2], x = t[3];

            // horizontal
            if(t[0] == t[2])
            {
                // go right
                if(x+1 < n && !grid[y][x+1] && !is_right_used(y, x+1))
                {
                    set_right(y, x+1);
                    q.push({y, x, y, x+1});
                }

                if(y+1 < n && !grid[y+1][x-1] && !grid[y+1][x])
                {
                    // go down
                    if(!is_right_used(y+1, x))
                    {
                        set_right(y+1, x);
                        q.push({y+1, x-1, y+1, x});
                    }

                    // rotate clockwise
                    if(!is_down_used(y+1, x-1))
                    {
                        set_down(y+1, x-1);
                        q.push({y, x-1, y+1, x-1});
                    }
                }
            }

            // vertical
            if(t[1] == t[3])
            {
                // go down
                if(y+1 < n && !grid[y+1][x] && !is_down_used(y+1, x))
                {
                    set_down(y+1, x);
                    q.push({y, x, y+1, x});
                }

                if(x+1 < n && !grid[y-1][x+1] && !grid[y][x+1])
                {
                    // go right
                    if(!is_down_used(y, x+1))
                    {
                        set_down(y, x+1);
                        q.push({y-1, x+1, y, x+1});
                    }

                    // rotate counter-clockwise
                    if(!is_right_used(y-1, x+1))
                    {
                        set_right(y-1, x+1);
                        q.push({y-1, x, y-1, x+1});
                    }
                }
            }
        }
        ++steps;
    }
    return -1;
}

int Solution::calculateMinimumHP(vector<vector<int>>& dungeon)
{
    /*
        The demons had captured the princess (P) and imprisoned her 
        in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms 
        laid out in a 2D grid. Our valiant knight (K) was initially positioned in the 
        top-left room and must fight his way through the dungeon to rescue the princess.

        The knight has an initial health point represented by a positive integer. 
        If at any point his health point drops to 0 or below, he dies immediately.

        Some of the rooms are guarded by demons, so the knight loses health (negative integers) 
        upon entering these rooms; other rooms are either empty (0’s) or contain magic orbs that 
        increase the knight’s health (positive integers).

        In order to reach the princess as quickly as possible, the knight decides to move 
        only rightward or downward in each step.

        Write a function to determine the knight’s minimum initial health so that 
        he is able to rescue the princess.

        For example, given the dungeon below, the initial health of the knight must be at least 7 
        if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN:

            [
                [-2 (K),-3,3],
                [-5,-10,1],
                [10,30,-5 (P)],
            ]
    */

    int m = (int)dungeon.size();
    int n = (int)dungeon[0].size();

    auto naive_dp = [&](int m, int n)
    {
        // dp[i][j] means minimum initial health to get (m, n) from (i, j), 
        // assume (i, j) is the start point.
        // dp[i][j] = max(1, min(dp[i + 1][j], hp[i][j + 1]) - dungeon[i][j])
        vector<vector<int>> dp(m+1, vector<int>(n+1, INT32_MAX));

        // precondition
        dp[m][n-1] = 1; dp[m-1][n] = 1;
        for(int i=m-1; i>=0; i--)
        {
            for(int j=n-1; j>=0; j--)
                dp[i][j] = std::max(1, std::min(dp[i+1][j], dp[i][j+1])) - dungeon[i][j];
        }
        return dp[0][0];
    };
    //return naive_dp(m, n);

    auto space_optimized_dp = [&](int m, int n)
    {
        vector<int> dp(n+1, INT32_MAX);
        dp[n-1] = 1;
        for(int i=m-1; i>=0; i--)
        {
            for(int j=n-1; j>=0; j--)
                dp[j] = std::max(1, std::min(dp[j], dp[j+1])) - dungeon[i][j];
        }
        return dp[0];
    };
    return space_optimized_dp(m, n);
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

void calculateMinimumHP_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.calculateMinimumHP(grid);
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

void minFallingSum_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minFallingSum(grid);
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

void minimumMoves_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.minimumMoves(grid);
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

    util::Log(logESSENTIAL) << "Running calculateMinimumHP tests:";
    TIMER_START(calculateMinimumHP);
    calculateMinimumHP_scaffold("[[-2,-3,3],[-5,-10,1],[10,30,-5]]", 7);
    TIMER_STOP(calculateMinimumHP);
    util::Log(logESSENTIAL) << "calculateMinimumHP using " << TIMER_MSEC(calculateMinimumHP) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minFallingSum tests:";
    TIMER_START(minFallingSum);
    minFallingSum_scaffold("[[1,2,3],[4,5,6],[7,8,9]]", 12);
    minFallingSum_scaffold("[[1],[2],[3]]", 6);
    TIMER_STOP(minFallingSum);
    util::Log(logESSENTIAL) << "minFallingSum using " << TIMER_MSEC(minFallingSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minimumMoves tests:";
    TIMER_START(minimumMoves);
    minimumMoves_scaffold("[[0,0,0,0,0,1],[1,1,0,0,1,0],[0,0,0,0,1,1],[0,0,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,0,0]]", 11);
    minimumMoves_scaffold("[[0,0,1,1,1,1],[0,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,1],[1,1,1,0,0,0]]", 9);
    TIMER_STOP(minimumMoves);
    util::Log(logESSENTIAL) << "minimumMoves using " << TIMER_MSEC(minimumMoves) << " milliseconds";
}
