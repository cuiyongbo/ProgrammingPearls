#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 542, 675, 934 */

class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix);
    int cutOffTree(vector<vector<int>>& forest);
    int shortestBridge(vector<vector<int>>& A);
};


/*
    Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell. The distance between two adjacent cells is 1.
    Note:
        The number of elements of the given matrix will not exceed 10,000.
        There are at least one 0 in the given matrix.
        The cells are adjacent in only four directions: up, down, left and right.
*/
vector<vector<int>> Solution::updateMatrix(vector<vector<int>>& matrix) {

{ // refined solution using bfs
    int rows = matrix.size();
    int columns = matrix[0].size();
    vector<vector<int>> ans(rows, vector<int>(columns, INT32_MAX));
    typedef std::pair<int, int> element_t;
    std::queue<element_t> q;
    std::set<element_t> visited;
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            if (matrix[i][j] == 0) {
                ans[i][j] = 0;
                q.emplace(i, j);
                visited.emplace(i, j);
            }
        }
    }
    int steps = 0;
    while (!q.empty()) {
        steps++;
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            for (auto d: directions) {
                int r = t.first + d.first;
                int c = t.second + d.second;
                if (r<0 || r>=rows || c<0 || c>=columns) {
                    continue;
                }
                if (matrix[r][c] == 1) {
                    ans[r][c] = std::min(steps, ans[r][c]);
                }
                auto p = std::make_pair(r, c);
                if (visited.count(p) == 0) {
                    q.push(p);
                    visited.insert(p);
                }
            }
        }
    }
    return ans;
}

{ // naive solution using bfs
    int rows = matrix.size();
    int columns = matrix[0].size();
    vector<vector<int>> ans(rows, vector<int>(columns, 0));
    auto bfs = [&] (int r, int c) {
        typedef std::pair<int, int> element_t;
        std::queue<element_t> q; q.emplace(r, c);
        std::set<element_t> visited; visited.emplace(r, c);
        int steps = 0;
        while (!q.empty()) {
            steps++;
            for (int k=q.size(); k!=0; --k) {
                auto t = q.front(); q.pop();
                for (auto d: directions) {
                    int r = t.first + d.first;
                    int c = t.second + d.second;
                    if (r<0 || r>=rows || c<0 || c>=columns) {
                        continue;
                    }
                    if (matrix[r][c] == 0) {
                        return steps;
                    }
                    auto p = std::make_pair(r, c);
                    if (visited.count(p) == 0) {
                        q.push(p);
                    }
                }
            }
        }        
        return 0;
    };
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            if (matrix[i][j] == 1) {
                ans[i][j] = bfs(i, j);
            }
        }
    }
    return ans;
}

{ // dp solution
    int rows = matrix.size();
    int columns = matrix[0].size();
    vector<vector<int>> dp(rows, vector<int>(columns, INT32_MAX-rows*columns));
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (matrix[r][c] == 0) {
                dp[r][c] = 0;
            } else {
                if (r>0) {
                    dp[r][c] = min(dp[r][c], dp[r-1][c]+1);
                }
                if (c>0) {
                    dp[r][c] = min(dp[r][c], dp[r][c-1]+1);
                }
            }
        }
    }
    for (int r=rows-1; r>=0; --r) {
        for (int c=columns-1; c>=0; --c) {
            if (r<rows-1) {
                dp[r][c] = min(dp[r][c], dp[r+1][c]+1);
            }
            if (c<columns-1) {
                dp[r][c] = min(dp[r][c], dp[r][c+1]+1);
            }
        }
    }
    return dp;
}

}


/*
    You are asked to cut off trees in a forest for a golf event. The forest is represented as a non-negative 2D map, in this map:
        0 represents the obstacle can’t be reached.
        1 represents the grass can be walked through.
        The place with number bigger than 1 represents a tree can be walked through, and this positive number represents the tree’s height.
    **You are asked to cut off all the trees in this forest in the order of tree’s height – always cut off the tree with lowest height first.**
    And after cutting, the original place has the tree will become a grass (value 1).

    You will start from the point (0, 0) and you should output the minimum steps you need to walk to cut off all the trees. 
    If you can’t cut off all the trees, return -1 in that situation.

    You are guaranteed that no two trees have the same height and there is at least one tree needs to be cut off.
    Hint: 
        1. fetch all the tree to be cut down, ordered by tree's height in ascending order (use priority_queue, or std::sort)
        2. find the minimus path between tree1 ans tree2 using bfs, and accumulate all length of all paths
*/
int Solution::cutOffTree(vector<vector<int>>& forest) {
    typedef std::pair<int, int> element_t;
    auto cmp = [&](const element_t& l, const element_t& r) {
        return forest[l.first][l.second] > forest[r.first][r.second];
    };
    std::priority_queue<element_t, vector<element_t>, decltype(cmp)> pq(cmp);

    int rows = forest.size();
    int columns = forest[0].size();
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            if (forest[i][j] != 0) {
                pq.emplace(i, j);
            }
        }
    }

    auto shortest_path_router = [&] (element_t start, element_t end) {
        int steps = 0;
        std::queue<element_t> q; q.push(start);
        std::set<element_t> visited; visited.insert(start);
        while (!q.empty()) {
            for (int k=q.size(); k!=0; --k) {
                auto t = q.front(); q.pop();
                if (t == end) {
                    return std::make_pair(true, steps);
                }
                for (auto d: directions) {
                    int r = t.first + d.first;
                    int c = t.second + d.second;
                    if (r<0 || r>=rows || c<0 || c>=columns || forest[r][c]==0) {
                        continue;
                    }
                    auto p = std::make_pair(r, c);
                    if (visited.count(p) == 0) {
                        visited.insert(p);
                        q.push(p);
                    }
                }
            }
            steps++;
        }
        return std::make_pair(false, 0);
    };

    int ans = 0;
    auto start = std::make_pair(0, 0);
    while (!pq.empty()) {
        auto end = pq.top(); pq.pop();
        auto p = shortest_path_router(start, end);
        if (!p.first) {
            return -1;
        }
        ans += p.second;
        start = end;
    }
    return ans;
}


/*
    In a given 2D binary array A, there are two islands.  
    (An island is a 4-directionally connected group of 1s not connected to any other 1s.)
    Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.
    Return the smallest number of 0s that must be flipped. (It is guaranteed that the answer is at least 1.)
    Hint:
        1. find 1-nodes corresponding one island
        2. starting from these nodes, continue bfs traversal until find a 1-node corresponding to the other island
*/
int Solution::shortestBridge(vector<vector<int>>& A) {

{ // naive solution
    int rows = A.size();
    int columns = A[0].size();
    typedef std::pair<int, int> element_t;
    std::set<element_t> island_s;
    std::queue<element_t> island_q;

    auto find_one_island = [&] (element_t start) {
        std::queue<element_t> q; q.push(start);
        std::set<element_t> visited; visited.insert(start);
        while (!q.empty()) {
            for (int k=q.size(); k!=0; --k) {
                auto t = q.front(); q.pop(); island_q.push(t);
                for (auto d: directions) {
                    int r = t.first + d.first;
                    int c = t.second + d.second;
                    if (r<0 || r>=rows || c<0 || c>=columns || A[r][c]==0) {
                        continue;
                    }
                    auto p = std::make_pair(r, c);
                    if (visited.count(p) == 0) {
                        visited.insert(p);
                        q.push(p);
                    }
                }
            }
        }
        island_s.swap(visited);
    };

    bool found = false;
    for (int i=0; i<rows && !found; ++i) {
        for (int j=0; j<columns; ++j) {
            if (A[i][j] == 1) {
                find_one_island({i, j});
                found = true;
                break;
            }
        }
    }

    int steps = 0;
    std::queue<element_t> q(island_q);
    std::set<element_t> visited(island_s);
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            if (A[t.first][t.second]==1 && island_s.count(t)==0) {
                return steps-1;
            }
            for (auto d: directions) {
                int r = t.first + d.first;
                int c = t.second + d.second;
                if (r<0 || r>=rows || c<0 || c>=columns) {
                    continue;
                }
                auto p = std::make_pair(r, c);
                if (visited.count(p) == 0) {
                    visited.insert(p);
                    q.push(p);
                }
            }
        }
        steps++;
    }
    return steps;
}

{ // refined version
    typedef pair<int, int> Coordinate;
    int rows = A.size();
    int columns = A[0].size();
    queue<Coordinate> q;
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    function<void(int, int)> dfs = [&] (int r, int c) {
        visited[r][c] = true;
        q.emplace(r, c);
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                continue;
            }
            if (A[nr][nc] == 0 || visited[nr][nc]) {
                continue;
            }
            dfs(nr, nc);
        }
    };

    bool stop = false;
    for (int r=0; r<rows && !stop; ++r) {
        for (int c=0; c<columns && !stop; ++c) {
            if (A[r][c] ==1) {
                dfs(r, c);
                stop = true;
            }   
        }
    }

    int steps = 0;
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto u = q.front(); q.pop();
            for (auto& d: directions) {
                int nr = u.first + d.first;
                int nc = u.second + d.second;
                if (nr<0 || nr>=rows || nc<0 || nc>=columns || visited[nr][nc]) {
                    continue;
                }
                if (A[nr][nc] == 1) {
                    return steps;
                }
                visited[nr][nc] = true;
                q.emplace(nr, nc);
            }
        }
        ++steps;
    }
    return -1;
}

}

void updateMatrix_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<int>> matrix = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.updateMatrix(matrix);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for (const auto& row: actual) {
            util::Log(logERROR) << numberVectorToString<int>(row);
        }
    }
}

void cutOffTree_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> forest = stringTo2DArray<int>(input);
    int actual = ss.cutOffTree(forest);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void shortestBridge_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> forest = stringTo2DArray<int>(input);
    int actual = ss.shortestBridge(forest);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
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