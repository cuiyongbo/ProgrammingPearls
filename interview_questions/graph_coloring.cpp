#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 785, 886, 1042*/
class Solution
{
public:
    bool isBipartite(vector<vector<int>>& graph);
    bool possibleBipartition(int N, vector<vector<int>>& dislikes);
    vector<int> gardenNoAdj(int N, vector<vector<int>>& paths);
};

bool Solution::isBipartite(vector<vector<int>>& graph)
{
    /*
        Given an undirected graph, return true if and only if it is bipartite.
        
        Recall that a graph is bipartite if we can split it's set of nodes into 
        two independent subsets A and B such that every edge in the graph has 
        one node in A and another node in B.
        
        The graph is given in the following form: graph[i] is a list of indexes j 
        for which the edge between nodes i and j exists.  Each node is an integer 
        between 0 and graph.length - 1. There are no self edges or parallel edges: 
        graph[i] does not contain i, and it doesn't contain any element twice.
    */

    if(graph.empty()) return true;
    int n = graph.size();
    vector<int> colors(n, 0);
    colors[0] = 1;
    queue<int> q; q.push(0);
    while(!q.empty())
    {
        auto u = q.front(); q.pop();

        for(auto v: graph[u])
        {
            if(colors[v] == 0)
            {
                q.push(v);
                colors[v] = colors[u] == 1 ? 2 : 1;
            }
            else if(colors[u] == colors[v])
            {
                return false;
            }
        }
    }
    return true;
}

bool Solution::possibleBipartition(int N, vector<vector<int>>& dislikes)
{
    /*
        Given a set of N people (numbered 1, 2, ..., N), we would like to split everyone into two groups of any size.
        Each person may dislike some other people, and they should not go into the same group. 
        Formally, if dislikes[i] = [a, b], it means it is not allowed to put the people numbered a and b into the same group.
        Return true if and only if it is possible to split everyone into two groups in this way.
    */

    vector<vector<int>> graph(N+1, vector<int>());
    for(auto& p: dislikes)
    {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    vector<int> colors(N+1, 0);
    colors[1] = 1;
    queue<int> q; q.push(1);
    while(!q.empty())
    {
        auto u = q.front(); q.pop();
        for(auto v: graph[u])
        {
            if(colors[u] == colors[v])
            {
                return false;
            }
            else if(colors[v] == 0)
            {
                colors[v] = colors[u] == 1 ? 2 : 1;
                q.push(v);
            }
        }
    }
   return true;
}

vector<int> Solution::gardenNoAdj(int N, vector<vector<int>>& paths)
{
    /*
        You have N gardens, labelled 1 to N. In each garden, you want to plant one of 4 types of flowers.
        paths[i] = [x, y] describes the existence of a bidirectional path from garden x to garden y.
        Also, there is no garden that has more than 3 paths coming into or leaving it.
        Your task is to choose a flower type for each garden such that, 
        for any two gardens connected by a path, they have different types of flowers.
        Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)-th garden.  
        The flower types are denoted 1, 2, 3, or 4.  It is guaranteed an answer exists.
    */

    vector<vector<int>> graph(N+1, vector<int>());
    for(auto& p: paths)
    {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    vector<int> colors(N+1, 0);
    for(int i=1; i<=N; ++i)
    {
        if(colors[i] != 0) continue;

        int mask = 0;
        for(auto j: graph[i]) mask |= (1 << colors[j]);
        for(int c=1; c<=4; ++c)
        {
            if(!(mask & (1<<c)))
            {
                colors[i] = c;
                break;
            }
        }
    }

    return vector<int>(colors.begin()+1, colors.end());
}

void isBipartite_scaffold(string input, bool expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    bool actual = ss.isBipartite(graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult<" << expectedResult << ">) passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult<" << expectedResult << ">) failed";
    }
}

void possibleBipartition_scaffold(int N, string input, bool expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    bool actual = ss.possibleBipartition(N, graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << N << ", " << input << ", expectedResult<" << expectedResult << ">) passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << N << ", " << input << ", expectedResult<" << expectedResult << ">) failed";
    }
}

void gardenNoAdj_scaffold(int N, string input, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(input);
    vector<int> actual = ss.gardenNoAdj(N, graph);
    vector<int> expected = stringToIntegerVector(expectedResult);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << N << ", " << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << N << ", " << input << ", " << expectedResult << ") failed";
        util::Log(logERROR) << "acutal: " << intVectorToString(actual) << ", expected: " << expectedResult;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isBipartite tests: ";
    TIMER_START(isBipartite);
    isBipartite_scaffold("[[1,3],[0,2],[1,3],[0,2]]", true);
    isBipartite_scaffold("[[1,2,3],[0,2],[1,3],[0,2]]", false);
    isBipartite_scaffold("[[1,2,3],[0,2],[0,1,3],[0,2]]", false);
    TIMER_STOP(isBipartite);
    util::Log(logESSENTIAL) << "isBipartite using " << TIMER_MSEC(isBipartite) << " milliseconds";

    util::Log(logESSENTIAL) << "Running possibleBipartition tests: ";
    TIMER_START(possibleBipartition);
    possibleBipartition_scaffold(4, "[[1,2],[1,3],[2,4]]", true);
    possibleBipartition_scaffold(3, "[[1,2],[1,3],[2,3]]", false);
    possibleBipartition_scaffold(5, "[[1,2],[2,3],[3,4],[4,5],[1,5]]", false);
    TIMER_STOP(possibleBipartition);
    util::Log(logESSENTIAL) << "possibleBipartition using " << TIMER_MSEC(possibleBipartition) << " milliseconds";

    util::Log(logESSENTIAL) << "Running gardenNoAdj tests: ";
    TIMER_START(gardenNoAdj);
    gardenNoAdj_scaffold(3, "[[1,2],[2,3],[3,1]]", "[1,2,3]");
    gardenNoAdj_scaffold(4, "[[1,2],[3,4]]", "[1,2,1,2]");
    gardenNoAdj_scaffold(4, "[[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]", "[1,2,3,4]");
    TIMER_STOP(gardenNoAdj);
    util::Log(logESSENTIAL) << "gardenNoAdj using " << TIMER_MSEC(gardenNoAdj) << " milliseconds";
}
