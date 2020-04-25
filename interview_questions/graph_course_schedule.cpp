#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 207, 210, 802 */

class Solution
{
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites);
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites);
    vector<int> eventualSafeNodes(vector<vector<int>>& graph);
};

bool Solution::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
    /*
        There are a total of n courses you have to take, labeled from 0 to n - 1.
        Some courses may have prerequisites, for example to take course 0 you have to first take course 1,
        which is expressed as a pair: [0,1]
        Given the total number of courses and a list of prerequisite pairs,
        is it possible for you to finish all courses?

        Hint: use dfs (coloring) to detect cycle in a graph.
    */

    vector<vector<int>> graph(numCourses, vector<int>());
    for(const auto& p : prerequisites)
    {
        if(p[0] == p[1]) return false;
        graph[p[0]].push_back(p[1]);
    }

    vector<int> colors(numCourses, 0);
    function<bool(int)> dfs = [&](int u)
    {
        colors[u] = 1; // visiting
        for(const auto& v: graph[u])
        {
            if(colors[v] == 0)
            { // unvisited
                if(!dfs(v))
                    return false;
            }
            else if(colors[v] == 1)
            {
                return false;
            }
        }
        colors[u] = 2; // visited
        return true;
    };

    for (int i = 0; i < numCourses; ++i)
    {
        if(colors[i] == 0)
        {
            if(!dfs(i))
                return false;
        }
    }
    return true;
}

vector<int> Solution::findOrder(int numCourses, vector<vector<int>>& prerequisites)
{
    /*
        There are a total of n courses you have to take, labeled from 0 to n - 1.
        Some courses may have prerequisites, for example to take course 0 you have
        to first take course 1, which is expressed as a pair: [0,1]

        Given the total number of courses and a list of prerequisite pairs,
        return the ordering of courses you should take to finish all courses.

        There may be multiple correct orders, you just need to return one of them.
        If it is impossible to finish all courses, return an empty array.
    */

    vector<int> ans;
    ans.reserve(numCourses);

    vector<vector<int>> graph(numCourses, vector<int>());
    for(const auto& p : prerequisites)
    {
        if(p[0] == p[1]) return ans;
        graph[p[0]].push_back(p[1]);
    }

    vector<int> colors(numCourses, 0);
    function<bool(int)> dfs = [&](int u)
    {
        colors[u] = 1; // visiting
        for(const auto& v: graph[u])
        {
            if(colors[v] == 0)
            { // unvisited
                if(!dfs(v))
                {
                    return false;
                }
            }
            else if(colors[v] == 1)
            {
                ans.clear();
                return false;
            }
        }
        ans.push_back(u);
        colors[u] = 2; // visited
        return true;
    };

    for (int i = 0; i < numCourses; ++i)
    {
        if(colors[i] == 0)
        {
            if(!dfs(i)) break;
        }
    }

    return ans;
}

vector<int> Solution::eventualSafeNodes(vector<vector<int>>& graph)
{
    /*
        In a directed graph, we start at some node and every turn,
        walk along a directed edge of the graph. If we reach a node that is terminal
        (that is, it has no outgoing directed edges), we stop.

        Now, say our starting node is eventually safe if and only if we must eventually
        walk to a terminal node. More specifically, there exists a natural number K so that
        for any choice of where to walk, we must have stopped at a terminal node in less than K steps.

        Which nodes are eventually safe? Return them as an array in sorted order.

        The directed graph has N nodes with labels 0, 1, ..., N-1, where N is the length of graph.
        The graph is given in the following form: graph[i] is a list of labels j such that (i, j)
        is a directed edge of the graph.
    */

    int nodeCount = graph.size();
    vector<int> colors(nodeCount, 0);

    set<int> candidate;
    function<bool(int)> dfs = [&](int u)
    {
        colors[u] = 1;
        for(const auto& v: graph[u])
        {
            if(colors[v] == 0)
            {
                if(!dfs(v))
                {
                    return false;
                }
            }
            else if(colors[v] == 1)
            {
                return false;
            }
        }
        colors[u] = 2;
        candidate.emplace(u);
        return true;
    };

    for (int i = 0; i < nodeCount; ++i)
    {
        if(colors[i] == 0)
            dfs(i);
    }
    return vector<int>(candidate.begin(), candidate.end());
}

void canFinish_scaffold(int numCourses, string input, bool expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray_t<int>(input);
    bool actual = ss.canFinish(numCourses, graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << numCourses << ", " << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << numCourses << ", " << input << "," << expectedResult  << ") failed";
    }
}

void findOrder_scaffold(int numCourses, string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray_t<int>(input);
    vector<int> actual = ss.findOrder(numCourses, graph);
    vector<int> expected = stringTo1DArray_t<int>(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << numCourses << ", " << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << numCourses << ", " << input << "," << expectedResult  << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", acutal: " << intVectorToString(actual);
    }
}

void eventualSafeNodes_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray_t<int>(input);
    vector<int> actual = ss.eventualSafeNodes(graph);
    vector<int> expected = stringTo1DArray_t<int>(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << "," << expectedResult << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", acutal: " << intVectorToString(actual);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running canFinish tests:";
    TIMER_START(canFinish);
    canFinish_scaffold(2, "[[1,0]]", true);
    canFinish_scaffold(2, "[[0,0]]", false);
    canFinish_scaffold(2, "[[1,0],[0,1]]", false);
    canFinish_scaffold(8, "[[1,0],[2,6],[1,7],[5,1],[6,4],[7,0],[0,5]]", false);
    canFinish_scaffold(8, "[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]]", true);
    TIMER_STOP(canFinish);
    util::Log(logESSENTIAL) << "canFinish using " << TIMER_MSEC(canFinish) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findOrder tests:";
    TIMER_START(findOrder);
    findOrder_scaffold(2, "[[1,0]]", "[0,1]");
    findOrder_scaffold(2, "[[0,0]]", "[]");
    findOrder_scaffold(2, "[[1,0],[0,1]]", "[]");
    findOrder_scaffold(8, "[[1,0],[2,6],[1,7],[5,1],[6,4],[7,0],[0,5]]", "[]");
    findOrder_scaffold(8, "[[1,0],[2,6],[1,7],[6,4],[7,0],[0,5]]", "[5,0,7,1,4,6,2,3]");
    TIMER_STOP(findOrder);
    util::Log(logESSENTIAL) << "findOrder using " << TIMER_MSEC(findOrder) << " milliseconds";

    util::Log(logESSENTIAL) << "Running eventualSafeNodes tests:";
    TIMER_START(eventualSafeNodes);
    eventualSafeNodes_scaffold("[[1,2],[2,3],[5],[0],[5],[],[]]", "[2,4,5,6]");
    eventualSafeNodes_scaffold("[[1],[2,4],[3],[2],[]]", "[4]");
    TIMER_STOP(eventualSafeNodes);
    util::Log(logESSENTIAL) << "eventualSafeNodes using " << TIMER_MSEC(eventualSafeNodes) << " milliseconds";
}
