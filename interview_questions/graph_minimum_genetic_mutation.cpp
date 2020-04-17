#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 443, 815 */

class Solution
{
public:
    int minMutation(string start, string end, vector<string>& bank);
    int numBusesToDestination(vector<vector<int>>& routes, int S, int T);

private:
    int numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T);
    int numBusesToDestination_dsu(vector<vector<int>>& routes, int S, int T);

};

int Solution::minMutation(string start, string end, vector<string>& bank)
{
    /*
        A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".

        Suppose we need to investigate about a mutation (mutation from “start” to “end”), where ONE mutation 
        is defined as ONE single character changed in the gene string.

        For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.

        Also, there is a given gene “bank”, which records all the valid gene mutations.
        A gene must be in the bank to make it a valid gene string.

        Now, given 3 things – start, end, bank, your task is to determine what is the minimum number 
        of mutations needed to mutate from “start” to “end”. If there is no such a mutation, return -1.

        Note:

            Starting point is assumed to be valid, so it might not be included in the bank.
            If multiple mutations are needed, all mutations during in the sequence must be valid.
            You may assume start and end string is not the same.
    */

    auto isValidMutation = [](const string& s, const string& b)
    {
        int count = 0;
        for(int i=0; i<(int)s.length(); ++i)
        {
            if(s[i] != b[i]) ++count;
        }
        return count == 1;
    };

    unordered_set<string> visited;
    visited.emplace(start);

    int steps = 0;
    queue<string> q;
    q.push(start);

    while(!q.empty())
    {
        int size = (int)q.size();
        for(int i=0; i<size; ++i)
        {
            auto s = std::move(q.front()); q.pop();
            if(s == end) return steps;


            for(const auto& b: bank)
            {
                if(visited.count(b) != 0) continue;
                if(!isValidMutation(s, b)) continue;
                visited.emplace(b);
                q.push(b);
            }
        }
        ++steps;
    }
    return -1;
}

int Solution::numBusesToDestination(vector<vector<int>>& routes, int S, int T)
{
    /*
        We have a list of bus routes. Each routes[i] is a bus route that the i-th bus repeats forever. 
        For example if routes[0] = [1, 5, 7], this means that the first bus (0-th indexed) travels 
        in the sequence 1->5->7->1->5->7->1->… forever.

        We start at bus stop S (initially not on a bus), and we want to go to bus stop T. 
        Travelling by buses only, what is the least number of buses we must take to reach 
        our destination? Return -1 if it is not possible.
    */

    return numBusesToDestination_bfs(routes, S, T);



}

int Solution::numBusesToDestination_dsu(vector<vector<int>>& routes, int S, int T)
{
    map<int, vector<int>> stationToBusMap;
    for(int i=0; i<(int)routes.size(); ++i)
    {
        for(const auto& n: routes[i])
            stationToBusMap[n].push_back(i);
    }
}

int Solution::numBusesToDestination_bfs(vector<vector<int>>& routes, int S, int T)
{
    map<int, vector<int>> stationToBusMap;
    for(int i=0; i<(int)routes.size(); ++i)
    {
        for(const auto& n: routes[i])
            stationToBusMap[n].push_back(i);
    }

    vector<bool> visited(routes.size(), false);

    int steps = 0;
    queue<int> q;
    q.push(S);
    while(!q.empty())
    {
        int size = (int)q.size();
        for(int i=0; i<size; ++i)
        {
            auto u = q.front(); q.pop();
            if(u == T)  return steps;

            for(const auto& r: stationToBusMap[u])
            {
                if(visited[r]) continue;
                visited[r] = true;

                for(const auto& v: routes[r])
                {
                    q.push(v);
                }
            }
        }
        ++steps;
    }

    return -1;
}


void minMutation_scaffold(string input1, string input2, string input3, int expectedResult)
{
    Solution ss;
    vector<string> bank = toStringArray(input3);
    int actual = ss.minMutation(input1, input2, bank);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << actual;
    }
}

void numBusesToDestination_scaffold(string input1, int input2, int input3, int expectedResult)
{
    Solution ss;
    vector<vector<int>> routes = stringTo2DArray(input1);
    int actual = ss.numBusesToDestination(routes, input2, input3);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actutal: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running minMutation tests:";
    TIMER_START(minMutation);
    minMutation_scaffold("AACCGGTT", "AACCGGTA", "[AACCGGTA]", 1);
    minMutation_scaffold("AACCGGTT", "AAACGGTA", "[AACCGGTA, AACCGCTA, AAACGGTA]", 2);
    minMutation_scaffold("AAAAACCC", "AACCCCCC", "[AAAACCCC, AAACCCCC, AACCCCCC]", 3);
    TIMER_STOP(minMutation);
    util::Log(logESSENTIAL) << "minMutation using " << TIMER_MSEC(minMutation) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numBusesToDestination tests:";
    TIMER_START(numBusesToDestination);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 5, -1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 1, 3, 1);
    numBusesToDestination_scaffold("[[1,3,7],[2,5,6]]", 2, 5, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 1, 6, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 2, 3, 2);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 6, 3, 1);
    numBusesToDestination_scaffold("[[1,2,7],[3,6,7]]", 7, 1, 1);
    TIMER_STOP(numBusesToDestination);
    util::Log(logESSENTIAL) << "numBusesToDestination using " << TIMER_MSEC(numBusesToDestination) << " milliseconds";

}
