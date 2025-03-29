#include "leetcode.h"

using namespace std;

/* leetcode: 980 */
class Solution {
public:
    int uniquePaths_980(vector<vector<int>>& grid);
};


/*
On a 2-dimensional grid, there are 4 types of squares:
        1 represents the starting square.  There is exactly one starting square.
        2 represents the ending square.  There is exactly one ending square.
        0 represents empty squares we can walk over.
    -1 represents obstacles that we cannot walk over.
Return the number of 4-directional paths from the starting square to the ending square, that **walk over every non-obstacle square(including start, end) exactly once.**
*/
int Solution::uniquePaths_980(vector<vector<int>>& grid) {
    // 1. find start, end position
    using Coordinate = std::pair<int, int>;
    int rows = grid.size();
    int columns = grid[0].size();
    Coordinate start, end;
    int non_obstacle_cnt = 0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            non_obstacle_cnt += (grid[r][c] != -1);
            if (grid[r][c] == 1) {
                start.first = r;
                start.second = c;
            }
            if (grid[r][c] == 2) {
                end.first = r;
                end.second = c;
            }
        }
    }
    // perform dfs search to find the number of paths from u to end which visit all non-obstacle nodes once and only once
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    function<int(Coordinate, int)> dfs = [&] (Coordinate u, int steps) {
        if (u==end || steps == non_obstacle_cnt) {
            return (u==end && steps==non_obstacle_cnt) ? 1 : 0;
        }
        int path_num = 0;
        visited[u.first][u.second] = true;
        for (auto& d: directions) {
            int nr = u.first + d.first;
            int nc = u.second + d.second;
            if (nr<0 || nr>=rows || 
                nc<0 || nc>=columns ||
                grid[nr][nc] == -1 || visited[nr][nc]) {
                continue;
            }
            path_num += dfs({nr, nc}, steps+1); // backtrace
        }
        visited[u.first][u.second] = false;
        return path_num;
    };
    return dfs(start, 1);
}


void uniquePaths_980_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    int actual = ss.uniquePaths_980(grid);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running uniquePaths_980 tests:");
    TIMER_START(uniquePaths_980);
    uniquePaths_980_scaffold("[[1,0,0,0],[0,0,0,0],[0,0,2,-1]]", 2);
    uniquePaths_980_scaffold("[[1,0,0,0],[0,0,0,0],[0,0,0,2]]", 4);
    uniquePaths_980_scaffold("[[0,1],[2,0]]", 0);
    TIMER_STOP(uniquePaths_980);
    SPDLOG_WARN("uniquePaths_980 tests use {} ms", TIMER_MSEC(uniquePaths_980));
}
