#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 200, 547, 695, 733, 827, 1162, 1020, 841, 1202 */

class Solution {
public:
    int numEnclaves(vector<vector<int>>& A);
    int numIslands(vector<vector<int>>& grid);
    int findCircleNum(vector<vector<int>>& M);
    int maxAreaOfIsland(vector<vector<int>>& grid);
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor);
    int largestIsland(vector<vector<int>>& grid);
    int maxDistance(vector<vector<int>>& grid);
    bool canVisitAllRooms(vector<vector<int>>& rooms);
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs);
};

string Solution::smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
    /*
        You are given a string s, and an array of pairs of indices in the string
        where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.
        You can swap the characters at any pair of indices in the given pairs any number of times.
        Return the lexicographically smallest string that s can be changed to after using the swaps.
    */

    int n = s.size();
    vector<vector<int>> graph(n, vector<int>());
    for (const auto& p: pairs) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    string buffer; buffer.reserve(n);
    vector<int> component; component.reserve(n);
    vector<int> visited(n, 0);
    function<void(int)> dfs = [&](int u) {
        visited[u] = 1;
        component.push_back(u);
        buffer.push_back(s[u]);
        for (const auto& v: graph[u]) {
            if (visited[v] == 0) {
                dfs(v);
            }
        }
        visited[u] = 2;
    };

    for (int u=0; u<n; ++u) {
        if (visited[u] == 0) {
            dfs(u);

            // got one connected component
            int i = 0;
            std::sort(buffer.begin(), buffer.end());
            std::sort(component.begin(), component.end());
            for (const auto& it: component) {
                s[it] = buffer[i++];
            }
            component.clear();
            buffer.clear();
        }
    }
    return s;
}

bool Solution::canVisitAllRooms(vector<vector<int>>& rooms) {
    /*
        There are N rooms and you start in room 0.
        Each room has a distinct number in 0, 1, 2, ..., N-1,
        and each room may have some keys to access the next room.

        Formally, each room i has a list of keys rooms[i], and each
        key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.
        A key rooms[i][j] = v opens the room with number v.

        Initially, all the rooms start locked (except for room 0). You can walk back 
        and forth between rooms freely. Return true if and only if you can enter every room.
    */

    set<int> unlockedRooms;
    function<void(int)> dfs = [&](int room) {
        for (const auto& r: rooms[room]) {
            if (unlockedRooms.count(r) == 0) {
                unlockedRooms.emplace(r);
                dfs(r);
            }
        }
    };
    unlockedRooms.emplace(0);
    dfs(0);
    return unlockedRooms.size() == rooms.size();
}

int Solution::numEnclaves(vector<vector<int>>& A) {
    /*
        Given a 2D array A, each cell is 0 (representing sea) or 1 (representing land)
        A move consists of walking from one land square 4-directionally 
        to another land square, or off the boundary of the grid.
        Return the number of land squares in the grid for which we cannot walk off
        the boundary of the grid in any number of moves.

        Example 1:
        Input: [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
        Output: 3
        Explanation: 
        There are three 1s that are enclosed by 0s, and one 1 that isn't enclosed because its on the boundary.

        Example 2:
        Input: [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
        Output: 0
        Explanation: 
        All 1s are either on the boundary or can reach the boundary.
        
        Note:
            1 <= A.length <= 500
            1 <= A[i].length <= 500
            0 <= A[i][j] <= 1
            All rows have the same size.
    */

    int rows = A.size();
    int columns = A[0].size();
    function<std::pair<bool, int>(int, int)> dfs = [&] (int r, int c) {
        if ((r<0 || r>rows)  || (c<0 || c> columns)) {
            return std::make_pair(true, 0);
        } else if (A[r][c] != 1) {
            return std::make_pair(false, 0);
        } else {
            A[r][c] = 2;
            int nodeCount = 1;
            bool isOffGrid = false;
            for (const auto& d: DIRECTIONS) {
                auto p = dfs(r + d[0], c + d[1]);
                isOffGrid |= p.first;
                nodeCount += p.second;
            }
            return std::make_pair(isOffGrid, nodeCount);
        }
    };

    int ans = 0;
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            if (A[i][j] == 1) {
                auto p = dfs(i, j);
                if (!p.first) {
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
    function<void(int, int)> dfs = [&](int row, int col) {
        if ((row < 0 || row >= rows) || (col < 0 || col >= columns)) {
            return;
        }
        if(grid[row][col] != 1) {
            return;
        }
        grid[row][col] = 2;
        for (auto& d: DIRECTIONS) {
            dfs(row+d[0], col+d[1]);
        }
    };

    int ans = 0;
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            if (grid[i][j] != 1) {
                continue;
            }
            ++ans;
            dfs(i, j);
        }
    }
    return ans;
}

int Solution::findCircleNum(vector<vector<int>>& M) {
    /*
        There are N students in a class. Some of them are friends, while some are not.
        Their friendship is transitive in nature. For example, if A is a direct friend of B,
        and B is a direct friend of C, then A is an indirect friend of C.
        And we defined a friend circle is a group of students who are direct or indirect friends.

        Given a N*N matrix M representing the friend relationship between students in the class.
        If M[i][j] = 1, then the ith and jth students are direct friends with each other, otherwise not.
        And you have to output the total number of friend circles among all the students.
    */

    return numIslands(M);
}

int Solution::maxAreaOfIsland(vector<vector<int>>& grid) {
    /*
        Given a non-empty 2D array grid of 0’s and 1’s, an island is a group of 1‘s (representing land)
        connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid
        are surrounded by water.

        Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)
    */

    int rows = grid.size();
    int columns = grid[0].size();
    function<int(int, int)> dfs = [&](int r, int c) {
        if ((r<0 || r>=rows) || (c<0 || c>=columns)) {
            return 0;
        } else if (grid[r][c] != 1) {
            return 0;
        } else {
            grid[r][c] = 2;

            int area = 1;
            for (const auto& d: DIRECTIONS) {
                area += dfs(r+d[0], c+d[1]);
            }
            return area;
        }
    };

    int ans = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            ans = std::max(dfs(i, j), ans);
        }
    }
    return ans;
}

vector<vector<int>> Solution::floodFill(vector<vector<int>>& grid, int sr, int sc, int newColor) {
    /*
        An image is represented by a 2D array of integers, each integer representing the pixel value
        of the image (from 0 to 65535).

        Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill,
        and a pixel value newColor, “flood fill” the image.

        To perform a “flood fill”, consider the starting pixel, plus any pixels connected 4-directionally
        to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally
        to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all
        of the aforementioned pixels with the newColor.

        At the end, return the modified image.
    */

    int rows = grid.size();
    int columns = grid[0].size();
    int originalColor = grid[sr][sc];
    function<void(int, int)> dfs = [&](int r, int c) {
        if (r<0 || r>=rows || c<0 || c>=columns) {
            return;
        } else if (grid[r][c] != originalColor) {
            return;
        } else {
            grid[r][c] = newColor;
            for (const auto& d: DIRECTIONS) {
                dfs(r+d[0], c+d[1]);
            }
        }
    };
    dfs(sr, sc);
    return grid;
}

int Solution::largestIsland(vector<vector<int>>& grid) {
    /*
        In a 2D grid of 0s and 1s, we change at most one 0 to a 1.
        After, what is the size of the largest island? (An island is a 4-directionally connected group of 1s).
    */

    int rows = grid.size();
    int columns = grid[0].size();

    int curIslandId = 0;
    map<int, int> islandAreaMap;
    map<Coordinate, int> coorToIslandIdMap;
    function<int(int, int)> dfs = [&](int r, int c) {
        if (r<0 || r>=rows || c<0 || c>=columns || grid[r][c] != 1) {
            return 0;
        } else {
            coorToIslandIdMap.emplace(Coordinate(r,c), curIslandId);
            grid[r][c] = 2;
            int area = 1;
            for (const auto& d: DIRECTIONS) {
                area += dfs(r+d[0], c+d[1]);
            }
            return area;
        }
    };

    int ans = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            if (grid[i][j] == 1) {
                islandAreaMap[curIslandId] = dfs(i, j);
                ans = std::max(ans, islandAreaMap[curIslandId]);
                ++curIslandId;
            }
        }
    }

    set<int> visited;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            if(grid[i][j] == 0) {
                visited.clear();
                for(const auto& d: DIRECTIONS) {
                    Coordinate c(i+d[0], j+d[1]);
                    const auto& it = coorToIslandIdMap.find(c);
                    if (it != coorToIslandIdMap.end()) {
                        visited.emplace(it->second);
                    }
                }

                int cur = 1;
                for (auto n: visited) {
                    cur += islandAreaMap[n];
                }
                ans = std::max(ans, cur);
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
    */

    int rows = grid.size();
    int columns = grid[0].size();
    queue<Coordinate> q;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            if (grid[i][j] == 1) {
                q.emplace(i, j);
            }
        }
    }

    int ans = -1;
    int steps = 0;
    while (!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; ++i) {
            const auto& c = q.front(); q.pop();
            if(grid[c.x][c.y] == 2) {
                // encounter a former water cell
                ans = steps;
            } 
            for (const auto& d: DIRECTIONS) {
                int x = c.x + d[0];
                int y = c.y + d[1];
                if(x<0 || x>=rows || y<0 || y>=columns || grid[x][y] != 0) {
                    continue;
                }
                // mark a water cell
                grid[x][y] = 2;
                q.emplace(x, y);
            }
        }
        ++steps;
    }
    return ans;
}

void numIslands_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.numIslands(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
    }
}

void findCircleNum_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    int actual = ss.findCircleNum(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult  << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult  << ", actual: " << actual;
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
        util::Log(logERROR) << "Case(" << input << "," << expectedResult  << ") failed";
    }
}

void smallestStringWithSwaps_scaffold(string input, string pairs, string expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(pairs);
    string actual = ss.smallestStringWithSwaps(input, graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << pairs << "," << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << pairs << "," << expectedResult  << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", actual: " << actual;
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
