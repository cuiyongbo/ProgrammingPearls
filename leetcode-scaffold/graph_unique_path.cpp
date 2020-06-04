#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 980 */

class Solution
{
public:
    int uniquePathsIII(vector<vector<int>>& grid);
};

int Solution::uniquePathsIII(vector<vector<int>>& grid)
{
    /*
        On a 2-dimensional grid, there are 4 types of squares:
            1 represents the starting square.  There is exactly one starting square.
            2 represents the ending square.  There is exactly one ending square.
            0 represents empty squares we can walk over.
            -1 represents obstacles that we cannot walk over.
        Return the number of 4-directional walks from the starting square to the ending square, 
        that walk over every non-obstacle square exactly once.
    */

    int rows = (int)grid.size();
    int columns = (int)grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));

    int sr=-1, sc=-1;
    for(int i=0; i<rows; ++i)
    {
        for(int j=0; j<columns; ++j)
        {
            if(grid[i][j] == 1)
            {
                sr=i; sc=j;
                break;
            }
        }
        if(sr >= 0) break;
    }

    auto isValidMove = [&](int r, int c)
    {
        return 0<=r && r<rows && 0<=c && c<columns && grid[r][c] != -1 && !visited[r][c]; 
    };

    auto isValidPath = [&]()
    {
        for(int i=0; i<rows; ++i)
        {
            for(int j=0; j<columns; ++j)
            {
                if(!visited[i][j] && grid[i][j] == 0) 
                    return false;
            }
        }
        return true;
    };

    int ans = 0;
    function<void(int, int)> backtrace = [&](int r, int c)
    {
        if(grid[r][c] == 2)
        {
            if(isValidPath()) ++ans;
            return;
        }

        visited[r][c] = true;
        for(const auto& m: DIRECTIONS)
        {
            int nr = r + m[1];
            int nc = c + m[0];
            if(isValidMove(nr, nc))
            {
                backtrace(nr, nc);
            }
        }
        visited[r][c] = false;
    };

    backtrace(sr, sc);

    return ans;    
}

void uniquePathsIII_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.uniquePathsIII(grid);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual:" << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running uniquePathsIII tests:";
    TIMER_START(uniquePathsIII);
    uniquePathsIII_scaffold("[[1,0,0,0],[0,0,0,0],[0,0,2,-1]]", 2);
    uniquePathsIII_scaffold("[[1,0,0,0],[0,0,0,0],[0,0,0,2]]", 4);
    uniquePathsIII_scaffold("[[0,1],[2,0]]", 0);
    TIMER_STOP(uniquePathsIII);
    util::Log(logESSENTIAL) << "uniquePathsIII using " << TIMER_MSEC(uniquePathsIII) << " milliseconds";
}
