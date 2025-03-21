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

if (0) { // disjoint set version solution
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
    int n = s.size();
    // build graph
    vector<vector<int>> graph(n);
    for (auto& p: pairs) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    // find all connected components
    vector<int> visited(n, 0);
    function<vector<int>(int u)> dfs = [&] (int u) {
        vector<int> path;
        path.push_back(u);
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                auto p = dfs(v);
                path.insert(path.end(), p.begin(), p.end());
            }
        }
        visited[u] = 2;
        return path;
    };
    vector<vector<int>> paths;
    for (int u=0; u<n; u++) {
        if (visited[u] == 0) {
            paths.push_back(dfs(u));
        }
    }
    // reoragnize s by sorting each component
    string ans = s;
    for (auto& p: paths) {
        std::sort(p.begin(), p.end());
        string tmp;
        for (auto i: p) {
            tmp.push_back(s[i]);
        }
        std::sort(tmp.begin(), tmp.end());
        for (int i=0; i<tmp.size(); i++) {
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
    std::function<int(int)> dfs = [&] (int u) {
        int count = 1;
        visited[u] = true;
        for (auto v: rooms[u]) {
            if (!visited[v]) {
                count += dfs(v);
            }
        }
        return count;
    };
    int visited_rooms = dfs(0);
    return visited_rooms == n;
}


int Solution::numEnclaves(vector<vector<int>>& grid) {
/*
    Given a 2D array grid, each cell is 0 (representing sea) or 1 (representing land).
    A move consists of walking from one land square 4-directionally to another land square, or off the boundary of the grid.
    Return the number of lands in the grid for which we cannot walk off the boundary of the grid in any number of moves.
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
    std::function<pair<bool, int>(int, int)> dfs = [&] (int r, int c) {
        int count = 1;
        visited[r][c] = true;
        bool in_bound = true;
        for (auto d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0||nr>=rows||nc<0||nc>=columns) {
                in_bound = false;
                continue;
            }
            if (grid[nr][nc]==1 && !visited[nr][nc]) {
                auto p = dfs(nr, nc);
                in_bound &= p.first;
                count += p.second;
            }
        }
        return make_pair(in_bound, count);
    };
    int ans = 0;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c]==1 && !visited[r][c]) {
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
    **A province is a group of directly or indirectly connected cities and no other cities outside of the group.**
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
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                continue;
            }
            if (grid[nr][nc] == 0) {
                continue;
            }
            if (!visited[nr][nc]) {
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

    int rows = grid.size();
    int columns = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(columns, false));
    std::function<int(int, int, int)> dfs = [&] (int r, int c, int island_id) {
        int area = 1; // count current node
        grid[r][c] = island_id; // mark node with current island
        visited[r][c] = true;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                continue;
            }
            if (grid[nr][nc] != 1) { // count only unknow island
                continue;
            }
            if (!visited[nr][nc]) {
                area += dfs(nr, nc, island_id);
            }
        }
        return area;
    };
    queue<pair<int, int>> zero_nodes;
    int island_id = 1;
    int max_island_area = 0;
    map<int, int> area_map; // island_id, area
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (grid[r][c] == 0) {
                zero_nodes.emplace(r, c);
            }
            if (grid[r][c] == 1 && !visited[r][c]) {
                island_id++;
                area_map[island_id] = dfs(r, c, island_id);
                max_island_area = max(max_island_area, area_map[island_id]);
            }
        }
    }
    int ans = max_island_area;
    while (!zero_nodes.empty()) {
        auto t = zero_nodes.front(); zero_nodes.pop();
        set<int> islands; // reachable islands
        for (auto& d: directions) {
            int nr = t.first + d.first;
            int nc = t.second + d.second;
            if (nr<0 || nr>=rows || nc<0 || nc>=columns) {
                continue;
            }
            if (grid[nr][nc] == 0) {
                continue;
            }
            islands.insert(grid[nr][nc]);
        }
        int area = 1; // count current zero node
        for (auto p: islands) {
            area += area_map[p];
        }
        ans = max(ans, area);
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
    queue<pair<int, int>> q;
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
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}


void findCircleNum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.findCircleNum(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void maxAreaOfIsland_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.maxAreaOfIsland(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void floodFill_scaffold(string input, int sr, int sc, int newColor, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.floodFill(graph, sr, sc, newColor);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, {}, {}, {}, expectedResult={}) passed", input, sr, sc, newColor, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, {}, expectedResult={}) failed, actual:", input, sr, sc, newColor, expectedResult);
        for (const auto& a: actual) {
            std::cout << numberVectorToString(a) << std::endl;
        }
    }
}

void largestIsland_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.largestIsland(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void maxDistance_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.maxDistance(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void numEnclaves_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.numEnclaves(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void canVisitAllRooms_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.canVisitAllRooms(graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

void smallestStringWithSwaps_scaffold(string input, string pairs, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(pairs);
    string actual = ss.smallestStringWithSwaps(input, graph);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input, expectedResult, actual);
    }
}

int main() {
    SPDLOG_WARN("Running numEnclaves tests:");
    TIMER_START(numEnclaves);
    numEnclaves_scaffold("[[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]", 3);
    numEnclaves_scaffold("[[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]", 0);
    TIMER_STOP(numEnclaves);
    SPDLOG_WARN("numEnclaves tests use {} ms", TIMER_MSEC(numEnclaves));

    SPDLOG_WARN("Running numIslands tests:");
    TIMER_START(numIslands);
    numIslands_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 1);
    numIslands_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 3);
    TIMER_STOP(numIslands);
    SPDLOG_WARN("numIslands tests use {} ms", TIMER_MSEC(numIslands));

    SPDLOG_WARN("Running findCircleNum tests:");
    TIMER_START(findCircleNum);
    findCircleNum_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 1);
    findCircleNum_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 3);
    findCircleNum_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 2);
    findCircleNum_scaffold("[[1,1,0],[1,1,0],[0,1,1]]", 1);
    findCircleNum_scaffold("[[1,0,0],[0,1,0],[0,0,1]]", 3);
    TIMER_STOP(findCircleNum);
    SPDLOG_WARN("findCircleNum tests use {} ms", TIMER_MSEC(findCircleNum));

    SPDLOG_WARN("Running maxAreaOfIsland tests:");
    TIMER_START(maxAreaOfIsland);
    maxAreaOfIsland_scaffold("[[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]]", 9);
    maxAreaOfIsland_scaffold("[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]", 4);
    maxAreaOfIsland_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 4);
    maxAreaOfIsland_scaffold("[[1,1,0],[1,1,0],[0,1,1]]", 6);
    maxAreaOfIsland_scaffold("[[0,0,0,0,0,0,0,0,0,0,0,0]]", 0);
    maxAreaOfIsland_scaffold("[[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0], [0,1,0,0,1,1,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]", 6);
    TIMER_STOP(maxAreaOfIsland);
    SPDLOG_WARN("maxAreaOfIsland tests use {} ms", TIMER_MSEC(maxAreaOfIsland));

    SPDLOG_WARN("Running floodFill tests:");
    TIMER_START(floodFill);
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 1, 1, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 0, 0, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 0, 1, 2, "[[2,2,2],[2,2,0],[2,0,1]]");
    floodFill_scaffold("[[1,1,1],[1,1,0],[1,0,1]]", 2, 2, 2, "[[1,1,1],[1,1,0],[1,0,2]]");
    TIMER_STOP(floodFill);
    SPDLOG_WARN("floodFill tests use {} ms", TIMER_MSEC(floodFill));

    SPDLOG_WARN("Running largestIsland tests:");
    TIMER_START(largestIsland);
    largestIsland_scaffold("[[1,0],[0,1]]", 3);
    largestIsland_scaffold("[[1,1],[0,1]]", 4);
    largestIsland_scaffold("[[1,1],[1,1]]", 4);
    largestIsland_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 6);
    TIMER_STOP(largestIsland);
    SPDLOG_WARN("largestIsland tests use {} ms", TIMER_MSEC(largestIsland));

    SPDLOG_WARN("Running maxDistance tests:");
    TIMER_START(maxDistance);
    maxDistance_scaffold("[[1,0],[0,1]]", 1);
    maxDistance_scaffold("[[1,1],[0,1]]", 1);
    maxDistance_scaffold("[[1,1],[1,1]]", -1);
    maxDistance_scaffold("[[0,0],[0,0]]", -1);
    maxDistance_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", 1);
    maxDistance_scaffold("[[1,0,1],[0,0,0],[1,0,1]]", 2);
    maxDistance_scaffold("[[0,0,0],[0,0,0],[0,0,1]]", 4);
    TIMER_STOP(maxDistance);
    SPDLOG_WARN("maxDistance tests use {} ms", TIMER_MSEC(maxDistance));

    SPDLOG_WARN("Running canVisitAllRooms tests:");
    TIMER_START(canVisitAllRooms);
    canVisitAllRooms_scaffold("[[0],[2],[3],[]]", false);
    canVisitAllRooms_scaffold("[[1],[2],[3],[]]", true);
    canVisitAllRooms_scaffold("[[1,3],[3,0,1],[2],[0]]", false);
    TIMER_STOP(canVisitAllRooms);
    SPDLOG_WARN("canVisitAllRooms tests use {} ms", TIMER_MSEC(canVisitAllRooms));

    SPDLOG_WARN("Running smallestStringWithSwaps tests:");
    TIMER_START(smallestStringWithSwaps);
    smallestStringWithSwaps_scaffold("dcab", "[[0,3],[1,2]]", "bacd");
    smallestStringWithSwaps_scaffold("dcab", "[[0,3],[1,2],[0,2]]", "abcd");
    smallestStringWithSwaps_scaffold("cba", "[[0,1],[1,2]]", "abc");
    TIMER_STOP(smallestStringWithSwaps);
    SPDLOG_WARN("smallestStringWithSwaps tests use {} ms", TIMER_MSEC(smallestStringWithSwaps));
}
