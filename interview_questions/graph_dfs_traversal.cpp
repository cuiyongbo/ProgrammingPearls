#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 200 */

class Solution
{
public:
    int numIslands(vector<vector<int>>& grid);
};


int Solution::numIslands(vector<vector<int>>& grid)
{
    /*
        Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
        An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
        You may assume all four edges of the grid are all surrounded by water.
    */

    const vector<vector<int>> directions {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows = grid.size();
    int columns = grid[0].size();
    function<void(int, int)> dfs = [&](int row, int col)
    {
        if(row < 0 || row >= rows)
            return;

        if(col < 0 || col >= columns)
            return;

        if(grid[row][col] == 0)
            return;

        grid[row][col] = 0;

        for(auto& d: directions)
        {
            dfs(row+d[0], col+d[1]);
        }
    };

    int ans = 0;
    for(int i=0; i<rows; ++i)
    {
        for(int j=0; j<columns; ++j)
        {
            if(grid[i][j] == 0)
                continue;

            ++ans;
            dfs(i, j);
        }
    }
    return ans;
}


void numIslands_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    int actual = ss.numIslands(graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running numIslands tests:";
    TIMER_START(numIslands);
    numIslands_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 1);
    numIslands_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 3);
    TIMER_STOP(numIslands);
    util::Log(logESSENTIAL) << "numIslands using " << TIMER_MSEC(numIslands) << " milliseconds";
}
