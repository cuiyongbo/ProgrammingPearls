#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 200, 547, 695, 733, 827, 841, 1020, 1162, 1202 */

class Solution {
public:
    int numIslands(vector<vector<int>>& grid);
    int findCircleNum(vector<vector<int>>& M);
    int maxAreaOfIsland(vector<vector<int>>& grid);
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor);
    int largestIsland(vector<vector<int>>& grid);
    bool canVisitAllRooms(vector<vector<int>>& rooms);
    int numEnclaves(vector<vector<int>>& A);
    int maxDistance(vector<vector<int>>& grid);
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs);
};

string Solution::smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
/*
    You are given a string s, and an array of pairs of indices in the string where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.
    You can swap the characters at any pair of indices in the given pairs any number of times.
    Return the lexicographically smallest string that s can be changed to after using the swaps.
    Hint:
        solution one: use to dfs to find connected components, then sort s.substring in each component
        solution two: use disjoint set to find connected components, then sort s.substring in each component
*/

{ // disjoint set version solution
    int sz = s.size();
    DisjointSet dsu(sz);
    for (auto& p: pairs) {
        dsu.unionFunc(p[0], p[1]);
    }
    map<int, vector<int>> group_map; // group id, node(s)
    for (int i=0; i<sz; ++i) {
        group_map[dsu.find(i)].push_back(i);
    }
    string ans = s;
    for (auto& p: group_map) {
        if (p.second.size() == 1) {
            continue;
        }
        string tmp;
        for (auto i: p.second) {
            tmp.push_back(s[i]);
        }
        sort(tmp.begin(), tmp.end());
        for (int i=0; i<tmp.size(); ++i) {
            ans[p.second[i]] = tmp[i];
        }
    }
    return ans;
}

{ // dfs version solution
    map<int, vector<int>> graph;
    for (auto& p: pairs) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    int sz = s.size();
    vector<int> visited(sz, 0);
    vector<int> path;
    vector<vector<int>> path_set;
    std::function<void(int)> dfs = [&] (int u) {
        visited[u] = 1;
        path.push_back(u);
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                dfs(v);
            }
        }
        visited[u] = 2;
    };
    for (int u=0; u<sz; ++u) {
        if (visited[u] == 0) {
            dfs(u);
            path_set.push_back(path);
            path.clear();
        }
    }
    string ans = s;
    for (auto& p : path_set) {
        std::sort(p.begin(), p.end());
        string tmp;
        for (auto i: p) {
            tmp.push_back(s[i]);
        }
        std::sort(tmp.begin(), tmp.end());
        for (int i=0; i<p.size(); ++i) {
            ans[p[i]] = tmp[i];
        }
    }
    return ans;
}

}

bool Solution::canVisitAllRooms(vector<vector<int>>& rooms) {
/*
    There are N rooms and you start in room 0. Each room has a distinct number in 0, 1, 2, ..., N-1, and each room may have some keys to access the next room.
    Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length. A key rooms[i][j] = v opens the room with number v.
    Initially, all the rooms start locked (except for room 0). You can walk back and forth between rooms freely. Return true if and only if you can enter every room.
*/
    int n = rooms.size();
    vector<bool> visited(n, false);
    std::function<void(int)> dfs = [&] (int u) {
        visited[u] = true;
        for (auto v: rooms[u]) {
            if (!visited[v]) {
                dfs(v);
            }
        }
    };
    dfs(0);
    return std::all_of(visited.begin(), visited.end(), [](bool i) {return i;});
}

int Solution::numEnclaves(vector<vector<int>>& grid) {
/*
    Given a 2D array grid, each cell is 0 (representing sea) or 1 (representing land).
    A move consists of walking from one land square 4-directionally to another land square, or off the boundary of the grid.
    Return the number of land squares in the grid for which we cannot walk off the boundary of the grid in any number of moves.
    Example 1:
        Input: [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
        Output: 3
        Explanation: There are three 1s that are enclosed by 0s, and one 1 that isn't enclosed because its on the boundary.
    Example 2:
        Input: [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
        Output: 0
        Explanation: All 1s are either on the boundary or can reach the boundary.
    Note:
        0 <= grid[i][j] <= 1
        All rows have the same size.
*/
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    function<pair<bool, int>(int, int)> dfs = [&] (int r, int c) {
        int node_count = 1;
        visited[r][c] = true;
        bool is_safe = true;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                is_safe = false;
                continue;
            } 
            if (grid[nr][nc]==1 && !visited[nr][nc]) {
                auto p = dfs(nr, nc);
                is_safe &= p.first;
                node_count += p.second;
            }  
        }
        return make_pair(is_safe, node_count);     
    };
    int ans = 0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1 && !visited[r][c]) {
                auto p = dfs(r, c);
                if (p.first) {
                    ans += p.second;
                }
            }
        }
    }
    return ans;
}

int Solution::numIslands(vector<vector<int>>& grid) {
/*
    Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
    You may assume all four edges of the grid are all surrounded by water.
*/
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    std::function<void(int, int)> dfs = [&] (int r, int c) {
        visited[r][c] = true;
        for (auto d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (0<=nr && nr<rows &&
                0<=nc && nc<columns &&
                !visited[nr][nc] &&
                grid[nr][nc] == 1) {
                dfs(nr, nc);
            }
        }
    };

    int ans = 0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1 && !visited[r][c]) {
                ++ans;
                dfs(r, c);
            }
        }
    }
    return ans;
}

int Solution::findCircleNum(vector<vector<int>>& grid) {
/*
    There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, 
    and city b is connected directly with city c, then city a is connected indirectly with city c.
    A province is a group of directly or indirectly connected cities and no other cities outside of the group.
    You are given an n x n matrix grid where grid[i][j] = 1 if the ith city and the jth city are directly connected, and grid[i][j] = 0 otherwise.
    Return the total number of provinces.
*/
    return numIslands(grid);
}

int Solution::maxAreaOfIsland(vector<vector<int>>& grid) {
/*
    Given a non-empty 2D array grid of 0’s and 1’s, an island is a group of 1‘s (representing land)
    connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
    Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)
*/
    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    std::function<int(int, int)> dfs = [&] (int r, int c) {
        int area = 1;
        visited[r][c] = true;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (0<=nr && nr<rows &&
                0<=nc && nc<columns &&
                !visited[nr][nc] &&
                grid[nr][nc] == 1) {
                area += dfs(nr, nc);
            }
        }
        return area;
    };

    int ans = 0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1 && !visited[r][c]) {
                ans = max(ans, dfs(r, c));
            }
        }
    }
    return ans;
}

vector<vector<int>> Solution::floodFill(vector<vector<int>>& grid, int sr, int sc, int newColor) {
/*
    An image is represented by a 2D array of integers, each integer representing the pixel value of the image (from 0 to 65535).
    Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, “flood fill” the image.
    To perform a “flood fill”, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, 
    plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on.
    Replace the color of all of the aforementioned pixels with the newColor. At the end, return the modified image.
*/
    int rows = grid.size();
    int columns = grid[0].size();
    int originalColor = grid[sr][sc];
    function<void(int, int)> dfs = [&] (int r, int c) {
        grid[r][c] = newColor;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (0<=nr && nr<rows &&
                0<=nc && nc<columns &&
                grid[nr][nc]==originalColor) {
                dfs(nr, nc);
            }
        }
    };
    dfs(sr, sc);
    return grid;
}

int Solution::largestIsland(vector<vector<int>>& grid) {
/*
    You are given an nxn binary matrix grid. You are allowed to change at most one 0 to be 1.
    Return the size of the largest island in grid after applying this operation. An island is a 4-directionally connected group of 1s.
*/
    typedef pair<int, int> Coordinate;
    int rows = grid.size();
    int columns = grid[0].size();
    int island_id = 0;
    map<Coordinate, int> coor_to_island_map; // coor, island_id
    map<int, int> island_area_map; // island_id, area
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    function<int(int, int)> dfs = [&] (int r, int c) {
        int area = 1;
        visited[r][c] = true;
        coor_to_island_map[{r, c}] = island_id;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (0<=nr && nr<rows &&
                0<=nc && nc<columns &&
                !visited[nr][nc] &&
                grid[nr][nc] == 1) {
                area += dfs(nr, nc);
            }
        }
        return area;
    };
    int ans = INT32_MIN;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c]==1 && !visited[r][c]) {
                island_area_map[island_id] = dfs(r, c);
                ans = max(ans, island_area_map[island_id]);
                ++island_id;
            }
        }
    }
    auto worker = [&] (int r, int c) {
        int area = 1;
        set<int> visited_islands;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (0<=nr && nr<rows &&
                0<=nc && nc<columns &&
                grid[nr][nc]==1) {
                Coordinate coor = make_pair(nr, nc);
                int island = coor_to_island_map[coor];
                if (visited_islands.count(island) == 0) {
                    visited_islands.insert(island);
                    area += island_area_map[island];
                }
            }
        }
        return area;
    };
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 0) {
                ans = max(ans, worker(r, c));
            }
        }
    }
    return ans;
}

int Solution::maxDistance(vector<vector<int>>& grid) {
/*
    Given an N x N grid containing only values 0 and 1, where 0 represents water and 1 represents land,
    find a water cell such that its distance to the nearest land cell is maximized and return the distance.
    The distance used in this problem is the Manhattan distance: the distance between two cells (x0, y0) 
    and (x1, y1) is |x0 - x1| + |y0 - y1|. If no land or water exists in the grid, return -1.
    Hint: launch a bfs traversal from all land cells, traverse ONLY water cells until no cell left, then we would find the answer
*/
    typedef pair<int, int> Coordinate;
    queue<Coordinate> q;
    int rows = grid.size();
    int columns = grid[0].size();
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 1) {
                q.emplace(r, c);
            }
        }
    }

    // all 0 or all 1
    if (q.empty() || q.size() == rows*columns) {
        return -1;
    }

    int steps = 0;
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    while (!q.empty()) {
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            for (auto& d: directions) {
                int nr = t.first + d.first;
                int nc = t.second + d.second;
                if (0<=nr && nr<rows &&
                    0<=nc && nc<columns &&
                    grid[nr][nc] == 0 && 
                    !visited[nr][nc]) {
                    q.emplace(nr, nc);
                    visited[nr][nc] = true;
                }
            }
        }
        ++steps;
    }
    return steps-1;
}

void numIslands_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.numIslands(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed, actual: " << actual;
    }
}

void findCircleNum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.findCircleNum(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed, actual: " << actual;
    }
}

void maxAreaOfIsland_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.maxAreaOfIsland(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

void floodFill_scaffold(string input, int sr, int sc, int newColor, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.floodFill(graph, sr, sc, newColor);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
    }
}

void largestIsland_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.largestIsland(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

void maxDistance_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.maxDistance(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

void numEnclaves_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.numEnclaves(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

void canVisitAllRooms_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.canVisitAllRooms(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << "," << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << "," << expectedResult  << ") failed, actual: " << actual;
    }
}

void smallestStringWithSwaps_scaffold(string input, string pairs, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(pairs);
    string actual = ss.smallestStringWithSwaps(input, graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << pairs << "," << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << pairs << "," << expectedResult  << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running numEnclaves tests:";
    TIMER_START(numEnclaves);
    numEnclaves_scaffold("[[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]", 3);
    numEnclaves_scaffold("[[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]", 0);
    TIMER_STOP(numEnclaves);
    util::Log(logESSENTIAL) << "numEnclaves using " << TIMER_MSEC(numEnclaves) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numIslands tests:";
    TIMER_START(numIslands);
    numIslands_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 1);
    numIslands_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 3);
    TIMER_STOP(numIslands);
    util::Log(logESSENTIAL) << "numIslands using " << TIMER_MSEC(numIslands) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findCircleNum tests:";
    TIMER_START(findCircleNum);
    findCircleNum_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 1);
    findCircleNum_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 3);
    findCircleNum_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 2);
    findCircleNum_scaffold("[[1,1,0],[1,1,0],[0,1,1]]", 1);
    findCircleNum_scaffold("[[1,0,0],[0,1,0],[0,0,1]]", 3);
    TIMER_STOP(findCircleNum);
    util::Log(logESSENTIAL) << "findCircleNum using " << TIMER_MSEC(findCircleNum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxAreaOfIsland tests:";
    TIMER_START(maxAreaOfIsland);
    maxAreaOfIsland_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 9);
    maxAreaOfIsland_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 4);
    maxAreaOfIsland_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 4);
    maxAreaOfIsland_scaffold("[[1,1,0],[1,1,0],[0,1,1]]", 6);
    maxAreaOfIsland_scaffold("[[0,0,0,0,0,0,0,0,0,0,0,0]]", 0);
    maxAreaOfIsland_scaffold("[[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0], [0,1,0,0,1,1,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]", 6);
    TIMER_STOP(maxAreaOfIsland);
    util::Log(logESSENTIAL) << "maxAreaOfIsland using " << TIMER_MSEC(maxAreaOfIsland) << " milliseconds";

    util::Log(logESSENTIAL) << "Running floodFill tests:";
    TIMER_START(floodFill);
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 1, 1, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 0, 0, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 0, 1, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 2, 2, 2, "[[1,1,1],[1,1,0],[1,0,2]]");
    TIMER_STOP(floodFill);
    util::Log(logESSENTIAL) << "floodFill using " << TIMER_MSEC(floodFill) << " milliseconds";

    util::Log(logESSENTIAL) << "Running largestIsland tests:";
    TIMER_START(largestIsland);
    largestIsland_scaffold("[[1,0],[0,1]]", 3);
    largestIsland_scaffold("[[1,1],[0,1]]", 4);
    largestIsland_scaffold("[[1,1],[1,1]]", 4);
    largestIsland_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 6);
    TIMER_STOP(largestIsland);
    util::Log(logESSENTIAL) << "largestIsland using " << TIMER_MSEC(largestIsland) << " milliseconds";

    util::Log(logESSENTIAL) << "Running maxDistance tests:";
    TIMER_START(maxDistance);
    maxDistance_scaffold("[[1,0],[0,1]]", 1);
    maxDistance_scaffold("[[1,1],[0,1]]", 1);
    maxDistance_scaffold("[[1,1],[1,1]]", -1);
    maxDistance_scaffold("[[0,0],[0,0]]", -1);
    maxDistance_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 1);
    maxDistance_scaffold("[[1,0,1],[0,0,0],[1,0,1]]", 2);
    maxDistance_scaffold("[[0,0,0],[0,0,0],[0,0,1]]", 4);
    TIMER_STOP(maxDistance);
    util::Log(logESSENTIAL) << "maxDistance using " << TIMER_MSEC(maxDistance) << " milliseconds";

    util::Log(logESSENTIAL) << "Running canVisitAllRooms tests:";
    TIMER_START(canVisitAllRooms);
    canVisitAllRooms_scaffold("[[0],[2],[3],[]]", false);
    canVisitAllRooms_scaffold("[[1],[2],[3],[]]", true);
    canVisitAllRooms_scaffold("[[1,3],[3,0,1],[2],[0]]", false);
    TIMER_STOP(canVisitAllRooms);
    util::Log(logESSENTIAL) << "canVisitAllRooms using " << TIMER_MSEC(canVisitAllRooms) << " milliseconds";

    util::Log(logESSENTIAL) << "Running smallestStringWithSwaps tests:";
    TIMER_START(smallestStringWithSwaps);
    smallestStringWithSwaps_scaffold("dcab", "[[0,3],[1,2]]", "bacd");
    smallestStringWithSwaps_scaffold("dcab", "[[0,3],[1,2],[0,2]]", "abcd");
    smallestStringWithSwaps_scaffold("cba", "[[0,1],[1,2]]", "abc");
    TIMER_STOP(smallestStringWithSwaps);
    util::Log(logESSENTIAL) << "smallestStringWithSwaps using " << TIMER_MSEC(smallestStringWithSwaps) << " milliseconds";    
}
