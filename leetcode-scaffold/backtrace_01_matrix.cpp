#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 542, 675, 934 */

class Solution 
{
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix);
    int cutOffTree(vector<vector<int>>& forest);
    int shortestBridge(vector<vector<int>>& A);

private:
    vector<vector<int>> updateMatrix_dp(vector<vector<int>>& matrix);
    vector<vector<int>> updateMatrix_bfs(vector<vector<int>>& matrix);
};

vector<vector<int>> Solution::updateMatrix(vector<vector<int>>& matrix)
{
    /*
        Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.
        The distance between two adjacent cells is 1.

        Note:
            The number of elements of the given matrix will not exceed 10,000.
            There are at least one 0 in the given matrix.
            The cells are adjacent in only four directions: up, down, left and right.
    */

    // return updateMatrix_bfs(matrix);
    return updateMatrix_dp(matrix);
}

vector<vector<int>> Solution::updateMatrix_bfs(vector<vector<int>>& matrix)
{
    int rows = (int)matrix.size();
    int columns = (int)matrix[0].size();
    vector<vector<int>> dist_table(rows, vector<int>(columns, INT32_MAX));

    auto bfs = [&](int c, int r)
    {
        queue<pair<int, int>> q; q.push({c, r});
        vector<vector<bool>> visited(rows, vector<bool>(columns, false));
        visited[r][c] = true;
        int steps = 0;
        while(!q.empty())
        {
            for(size_t k=q.size(); k != 0; --k)
            {
                auto p = q.front(); q.pop();
                if(matrix[p.second][p.first] == 1) 
                    dist_table[p.second][p.first] = std::min(dist_table[p.second][p.first], steps);

                for(const auto& d: DIRECTIONS)
                {
                    int x = p.first + d[0];
                    int y = p.second + d[1];
                    if( 0<=x && x<columns &&
                        0<=y && y<rows &&
                        !visited[y][x])
                    {
                        q.push({x, y});
                        visited[y][x] = true;
                    }
                }
            }
            ++steps;
        }
    };

    for(int j=0; j<rows; j++)
    {
        for(int i=0; i<columns; i++)
        {
            if(matrix[j][i] == 0) 
            {
                dist_table[j][i] = 0;
                bfs(i, j);
            }
        }
    }
    
    return dist_table;
}

vector<vector<int>> Solution::updateMatrix_dp(vector<vector<int>>& matrix)
{
    int rows = (int)matrix.size();
    int columns = (int)matrix[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns, INT32_MAX-rows*columns));

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            if(matrix[i][j]) 
            {
                if(i > 0) dp[i][j] = std::min(dp[i][j], dp[i-1][j]+1);
                if(j > 0) dp[i][j] = std::min(dp[i][j], dp[i][j-1]+1);
            }
            else
            {
                dp[i][j] = 0;
            }
        }
    }

    for(int i=rows-1; i>=0; i--)
    {
        for(int j=columns-1; j>=0; j--)
        {
            if(i<rows-1) dp[i][j] = std::min(dp[i][j], dp[i+1][j]+1);
            if(j<columns-1) dp[i][j] = std::min(dp[i][j], dp[i][j+1]+1);
        }
    }
    
    return dp;
}

int Solution::cutOffTree(vector<vector<int>>& forest)
{
    /*
        You are asked to cut off trees in a forest for a golf event. The forest is represented as a non-negative 2D map, 
        in this map:
            0 represents the obstacle can’t be reached.
            1 represents the ground can be walked through.
            The place with number bigger than 1 represents a tree can be walked through, 
            and this positive number represents the tree’s height.
        You are asked to cut off all the trees in this forest in the order of tree’s height – always 
        cut off the tree with lowest height first. And after cutting, the original place has the tree 
        will become a grass (value 1).

        You will start from the point (0, 0) and you should output the minimum steps you need to walk 
        to cut off all the trees. If you can’t cut off all the trees, output -1 in that situation.

        You are guaranteed that no two trees have the same height and there is at least one tree needs to be cut off.
    */

    struct Tree
    {
        int x, y;
        int height;

        Tree(int a, int b, int h): x(a), y(b), height(h) {}

        bool operator<(const Tree& rhs) const
        {
            return std::tie(height, x, y) < std::tie(rhs.height, rhs.x, rhs.y);
        }

        bool operator==(const Tree& rhs) const
        {
            return std::tie(x, y) == std::tie(rhs.x, rhs.y);
        }
    };

    vector<Tree> destinations;
    set<Tree> blocks;
    int rows = (int)forest.size();
    int columns = (int)forest[0].size();
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            if(forest[i][j] > 1)
                destinations.emplace_back(i, j, forest[i][j]);
            else if(forest[i][j] == 0)
                blocks.emplace(i, j, 0);
        }
    }

    std::sort(destinations.begin(), destinations.end());

    auto bfs = [&](Tree s, Tree t)
    {
        vector<vector<bool>> visited(rows, vector<bool>(columns, false));
        visited[s.y][s.x] = true;
        queue<Tree> q; q.push(s);
        int steps = 0;
        while(!q.empty())
        {
            for(size_t s=q.size(); s!=0; s--)
            {
                auto p = q.front(); q.pop();
                if(p==t) return steps;
                for(const auto& d: DIRECTIONS)
                {
                    int x = p.x + d[0];
                    int y = p.y + d[1];
                    if( 0<=x && x<columns &&
                        0<=y && y<rows &&
                        blocks.count(Tree(x, y, 0)) == 0 &&
                        !visited[y][x])
                    {
                        visited[y][x] = true;
                        q.emplace(x, y, forest[y][x]);
                    }
                }
            }
            ++steps;
        }
        return -1;
    };

    int ans = 0;
    Tree start(0, 0, 0);
    for(const auto& t: destinations)
    {
        int len = bfs(start, t);
        if(len == -1) return -1;
        ans += len;
        start = t;
    }
    return ans;
}

int Solution::shortestBridge(vector<vector<int>>& A)
{
    /*
        In a given 2D binary array A, there are two islands.  
        (An island is a 4-directionally connected group of 1s not connected to any other 1s.)
        Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.
        Return the smallest number of 0s that must be flipped. (It is guaranteed that the answer is at least 1.)
    */

    int rows = (int)A.size();
    int columns = (int)A[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    queue<pair<int, int>> q;
    function<void(int, int)> dfs = [&](int r, int c)
    {
        visited[r][c] = true;
        q.emplace(c, r);
        for(const auto& d: DIRECTIONS)
        {
            int x = c + d[0];
            int y = r + d[1];
            if( 0<=x && x<columns &&
                0<=y && y<rows &&
                A[y][x] == 1 &&
                !visited[y][x])
            {
                visited[y][x] = true;
                dfs(y, x);
            }
        }        
    };

    // find one connected component as sources
    bool found = false;
    for(int i=0; i<rows && !found; i++)
    {
        for(int j=0; j<columns && !found; j++)
        {
            if(A[i][j] == 1)
            {
                found = true;
                dfs(i, j);
            }
        }
    }

    int steps = 0;
    while(!q.empty())
    {
        for(size_t s=q.size(); s!=0; s--)
        {
            auto p = q.front(); q.pop();
            for(const auto& d: DIRECTIONS)
            {
                int x = p.first + d[0];
                int y = p.second + d[1];
                if( 0<=x && x<columns &&
                    0<=y && y<rows &&
                    !visited[y][x])
                {
                    if(A[y][x] == 1) return steps;
                    visited[y][x] = true;
                    q.emplace(x, y);
                }
            }
        }
        ++steps;
    }
    return 0;
}

void updateMatrix_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.updateMatrix(matrix);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& row: actual)
        {
            util::Log(logERROR) << numberVectorToString<int>(row);
        }
    }
}

void cutOffTree_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> forest = stringTo2DArray<int>(input);
    int actual = ss.cutOffTree(forest);
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

void shortestBridge_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> forest = stringTo2DArray<int>(input);
    int actual = ss.shortestBridge(forest);
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

    util::Log(logESSENTIAL) << "Running updateMatrix tests: ";
    TIMER_START(updateMatrix);
    updateMatrix_scaffold("[[0,0,0],[0,1,0],[0,0,0]]", "[[0,0,0],[0,1,0],[0,0,0]]");
    updateMatrix_scaffold("[[0,0,0],[0,1,0],[1,1,1]]", "[[0,0,0],[0,1,0],[1,2,1]]");
    TIMER_STOP(updateMatrix);
    util::Log(logESSENTIAL) << "updateMatrix using " << TIMER_MSEC(updateMatrix) << " milliseconds";

    util::Log(logESSENTIAL) << "Running cutOffTree tests: ";
    TIMER_START(cutOffTree);
    cutOffTree_scaffold("[[1,2,3],[0,0,4],[7,6,5]]", 6);
    cutOffTree_scaffold("[[1,2,3],[0,0,0],[7,6,5]]", -1);
    cutOffTree_scaffold("[[2,3,4],[0,0,5],[8,7,6]]", 6);
    cutOffTree_scaffold("[[3,4,5],[0,0,6],[2,8,7]]", 17);
    TIMER_STOP(cutOffTree);
    util::Log(logESSENTIAL) << "cutOffTree using " << TIMER_MSEC(cutOffTree) << " milliseconds";

    util::Log(logESSENTIAL) << "Running shortestBridge tests: ";
    TIMER_START(shortestBridge);
    shortestBridge_scaffold("[[0,1],[1,0]]", 1);
    shortestBridge_scaffold("[[0,1,0],[0,0,0],[0,0,1]]", 2);
    shortestBridge_scaffold("[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]", 1);
    TIMER_STOP(shortestBridge);
    util::Log(logESSENTIAL) << "shortestBridge using " << TIMER_MSEC(shortestBridge) << " milliseconds";

}