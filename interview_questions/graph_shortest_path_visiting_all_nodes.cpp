#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 847, 864 */

class Solution
{
public:
    int shortestPathLength(vector<vector<int>>& graph);
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

    // for debug
    for(int i=0; i<nodeCount; ++i)
    {
        cout << numberVectorToString(distTable[i]) << endl;
    }

    auto tripLengthForPlan = [&](vector<int>& nodeOrder)
    {
        int length = 0;
        for(int i=1; i<nodeCount; ++i)
        {
            if(distTable[nodeOrder[i-1]][nodeOrder[i]] == INT32_MAX)
                return INT32_MAX;

            length += distTable[nodeOrder[i-1]][nodeOrder[i]];
        }
        return length;
    };

    // Time Limit Exceeded for brute force method
    int ans = INT32_MAX;
    vector<int> nodeOrder(nodeCount, 0);
    vector<int> plan = nodeOrder;
    std::iota(nodeOrder.begin(), nodeOrder.end(), 0);
    do
    {
        int len = tripLengthForPlan(nodeOrder);
        if(len < ans)
        {
            plan = nodeOrder;
            ans = len;
        }
    } while (std::next_permutation(nodeOrder.begin(), nodeOrder.end()));
    
    // for debug
    cout << "optimal plan: " << numberVectorToString(plan) << endl;

    return ans;
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


}
