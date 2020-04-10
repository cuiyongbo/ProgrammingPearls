#include "leetcode.h"
#include "util/trip_farthest_insertion.hpp"

using namespace std;
using namespace osrm;

/* leetcode exercises: 847, 864, 1298 */

#define DEBUG_VERBOSITY

class Solution
{
public:
    int shortestPathLength(vector<vector<int>>& graph);
    int shortestPathAllKeys(vector<string>& grid);

private:
    int bruteForceTrip(vector<vector<int>>& distTable, int nodeCount);
};

int Solution::shortestPathLength(vector<vector<int>>& graph)
{
    /*
        An undirected, connected graph of N nodes (labeled 0, 1, 2, ..., N-1) is given as graph.
        graph.length = N, and j != i is in the list graph[i] exactly once, if and only if nodes i and j are connected.
        Return the length of the shortest path that visits every node.
        You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.

        Hint: Traveling Salesman Problem 
    */

    int nodeCount = (int)graph.size();
    vector<vector<int>> distTable(nodeCount, vector<int>(nodeCount, INT32_MAX));
    for(int u=0; u<nodeCount; ++u)
    {
        distTable[u][u] = 0;
        for(const auto& v: graph[u])
        {
            distTable[u][v] = 1;
            distTable[v][u] = 1;
        }
    }

    for(int k=0; k<nodeCount; ++k)
    {
        for(int i=0; i<nodeCount; ++i)
        {
            for(int j=0; j<nodeCount; ++j)
            {
                if(distTable[i][k] != INT32_MAX && distTable[k][j] != INT32_MAX)
                {
                    distTable[i][j] = std::min(distTable[i][j], distTable[i][k]+distTable[k][j]);
                }
            }
        }
    }

    int maxWeight = INT32_MIN;
    for(const auto& r: distTable)
    {
        auto it = std::max_element(r.begin(), r.end());
        maxWeight = std::max(maxWeight, *it);
    }

    BOOST_ASSERT_MSG(maxWeight != INT32_MAX, "graph is not a strongly connected component");

#if defined(DEBUG_VERBOSITY)
    for(int i=0; i<nodeCount; ++i)
    {
        cout << numberVectorToString(distTable[i]) << endl;
    }
#endif

    // Taken from OSRM project
    if(nodeCount < 10)
    {
        // Time Limit Exceeded
        return bruteForceTrip(distTable, nodeCount);
    }
    else
    {
        vector<int> nodeOrder = scaffold::farthestInsertionTrip(nodeCount, distTable);

#if defined(DEBUG_VERBOSITY)
        cout << "Optimal plan: " << numberVectorToString(nodeOrder) << endl;
#endif
        int length = 0;
        for(int i=1; i<nodeCount; ++i)
        {
            length += distTable[nodeOrder[i-1]][nodeOrder[i]];
        }
        return length;
    }
}

int Solution::bruteForceTrip(vector<vector<int>>& distTable, int nodeCount)
{
    auto tripLengthForPlan = [&](vector<int>& nodeOrder, int minLenth)
    {
        int length = 0;
        for(int i=1; i<nodeCount; ++i)
        {
            if(distTable[nodeOrder[i-1]][nodeOrder[i]] == INT32_MAX)
                return INT32_MAX;

            length += distTable[nodeOrder[i-1]][nodeOrder[i]];
            if(length >= minLenth) break;
        }
        return length;
    };

    int ans = INT32_MAX;
    vector<int> nodeOrder(nodeCount, 0);
    vector<int> plan = nodeOrder;
    std::iota(nodeOrder.begin(), nodeOrder.end(), 0);
    do
    {
        int len = tripLengthForPlan(nodeOrder, ans);
        if(len < ans)
        {
            plan = nodeOrder;
            ans = len;
        }
    } while (std::next_permutation(nodeOrder.begin(), nodeOrder.end()));
    
#if defined(DEBUG_VERBOSITY)
    cout << "optimal plan: " << numberVectorToString(plan) << endl;
#endif
    return ans;    
}

int Solution::shortestPathAllKeys(vector<string>& grid)
{
    /*
        We are given a 2-dimensional grid. "." is an empty cell, "#" is a wall, "@" is the starting point, 
        ("a", "b", …) are keys, and ("A", "B", …) are locks.

        We start at the starting point, and one move consists of walking one space in one of the 4 cardinal directions.
        We cannot walk outside the grid, or walk into a wall.  If we walk over a key, we pick it up.
        We can’t walk over a lock unless we have the corresponding key.

        For some 1 <= K <= 6, there is exactly one lowercase and one uppercase letter of the first K letters 
        of the English alphabet in the grid. This means that there is exactly one key for each lock, 
        and one lock for each key; and also that the letters used to represent the keys and locks 
        were chosen in the same order as the English alphabet.

        Return the lowest number of moves to acquire all keys.  If it’s impossible, return -1.
    */

    Coordinate coors[128];
    int rows = grid.size();
    int columns = grid[0].size();
    int keys = 0;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; ++j)
        {
            if(grid[i][j] == '@' 
                || 'a' <= grid[i][j] && grid[i][j] <= 'f' 
                || 'A' <= grid[i][j] && grid[i][j] <= 'F')
            {
                keys++;
                coors[grid[i][j]].x = j;
                coors[grid[i][j]].y = i;
            }
        }
    }

    keys = (keys-1)/2;
    auto route_length = [&] (vector<int>& route, int min_len)
    {
        int length = 0;
        Coordinate start = coors['@'];
        vector<bool> visited(keys, false);
        for(const auto& u: route)
        {
            queue<Coordinate> q;
            q.push(start);
            

            visited[u] = true;
        }

        return INT32_MAX;
    };

    vector<int> plan;
    int min_len = INT32_MAX;

    vector<int> route(keys);
    std::iota(route.begin(), route.end(), 0);
    do
    {
        int len = route_length(route, min_len);
        if(len < min_len)
        {
            len = min_len;
            plan = route;
        }
    } while (std::next_permutation(route.begin(), route.end()));

#if defined(DEBUG_VERBOSITY)
    cout << numberVectorToString(plan) << endl;
#endif
    return min_len == INT32_MAX ? -1 : min_len;
}

void shortestPathLength_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    int actual = ss.shortestPathLength(graph);
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

void shortestPathAllKeys_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<string> graph = toStringArray(input);
    int actual = ss.shortestPathAllKeys(graph);
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

    util::Log(logESSENTIAL) << "Running shortestPathLength tests:";
    TIMER_START(shortestPathLength);
    shortestPathLength_scaffold("[[1,2,3],[0],[0],[0]]", 4);
    shortestPathLength_scaffold("[[1],[0,2,4],[1,3,4],[2],[1,2]]", 4);
    shortestPathLength_scaffold("[[2,6],[2,3],[0,1],[1,4,5,6,8],[3,9,7],[3],[3,0],[4],[3],[4]]", 12);
    TIMER_STOP(shortestPathLength);
    util::Log(logESSENTIAL) << "shortestPathLength using " << TIMER_MSEC(shortestPathLength) << " milliseconds";

    util::Log(logESSENTIAL) << "Running shortestPathAllKeys tests:";
    TIMER_START(shortestPathAllKeys);
    shortestPathAllKeys_scaffold("[@.a.#, ###.#, b.A.B]", 8);
    shortestPathAllKeys_scaffold("[@..aA, ..B#., ....b]", 6);
    TIMER_STOP(shortestPathAllKeys);
    util::Log(logESSENTIAL) << "shortestPathAllKeys using " << TIMER_MSEC(shortestPathAllKeys) << " milliseconds";


}
