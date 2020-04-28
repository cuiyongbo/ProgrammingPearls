#include "leetcode.h"

using namespace std;
using namespace osrm;

class Solution
{
public:
    vector<vector<int>> tarjan_alg(int n, vector<vector<int>>& edges);
};

vector<vector<int>> Solution::tarjan_alg(int node_count, vector<vector<int>>& edges)
{
    /*
        Given a directed graph with nodes labelled 0 to `node_count-1`, edge `[u, v]`
        means there is a directed edge connected node u and v, where u is a parent of child v.
        find SCCs in a directed graph using tarjan algorithm.
    */

    vector<vector<int>> graph(node_count);
    for(const auto& p: edges)
    {
        graph[p[0]-1].push_back(p[1]-1);
    }

    vector<int> discovered_time(node_count, -1);
    vector<int> low_time(node_count, -1);
    vector<bool> on_stack(node_count, false);

    int tick = 0;
    stack<int> st;
    vector<vector<int>> scc_list;

    function<void(int)> dfs = [&](int u)
    {
        discovered_time[u] = tick;
        low_time[u] = tick;
        st.push(u);
        on_stack[u] = true;
        tick++;

        for(const auto& v: graph[u])
        {
            if(discovered_time[v] == -1)
            {
                // unvisited
                dfs(v);
                low_time[u] = std::min(low_time[u], low_time[v]);
            }
            else if(on_stack[v])
            {
                // already discovered
                low_time[u] = std::min(low_time[u], discovered_time[v]);
            }
        } 

        if(discovered_time[u] == low_time[u])
        {
            int v = u;
            vector<int> scc;
            do
            {
                v = st.top();
                st.pop();
                on_stack[v] = false;
                scc.push_back(v+1);
            } while (u != v);
            scc_list.push_back(scc);            
        }
    };

    for(int u=0; u < node_count; ++u)
    {
        if(discovered_time[u] == -1)
            dfs(u);
    }

    for(auto& scc: scc_list) std::sort(scc.begin(), scc.end());
    std::sort(scc_list.begin(), scc_list.end());

    return scc_list;
}

void tarjan_alg_scaffold(int input1, string input2, string expectedResult)
{
    Solution ss;
    vector<vector<int>> edges = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.tarjan_alg(input1, edges);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s);
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running tarjan_alg tests:";
    TIMER_START(tarjan_alg);
    tarjan_alg_scaffold(8, "[[1,3],[2,1],[3,2],[3,4],[4,5],[5,6],[6,7],[7,8],[8,5]]", "[[1,2,3],[4],[5,6,7,8]]");
    tarjan_alg_scaffold(9, "[[1,2],[2,1],[2,5],[3,4],[4,3],[4,5],[5,6],[6,7],[7,8],[8,6],[8,9]]", "[[1,2],[3,4],[5],[6,7,8],[9]]");
    TIMER_STOP(tarjan_alg);
    util::Log(logESSENTIAL) << "tarjan_alg using " << TIMER_MSEC(tarjan_alg) << " milliseconds";
}
