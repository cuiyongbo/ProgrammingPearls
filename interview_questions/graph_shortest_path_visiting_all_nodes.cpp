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
    */


    return 0;
}

void shortestPathLength_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    int actual = ss.shortestPathLength(graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed\n";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed\n";
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
    TIMER_STOP(shortestPathLength);
    util::Log(logESSENTIAL) << "shortestPathLength using " << TIMER_MSEC(shortestPathLength) << " milliseconds";


}
