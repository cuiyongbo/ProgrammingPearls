#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 332*/

class Solution
{
public:
    vector<string> findItinerary(vector<vector<string>>& tickets);
};

vector<string> Solution::findItinerary(vector<vector<string>>& tickets)
{
    /*
        Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], 
        reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK.
        Thus, the itinerary must begin with JFK.

        Note:

            If there are multiple valid itineraries, you should return the itinerary that has the smallest 
            lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a 
            smaller lexical order than ["JFK", "LGB"].
            All airports are represented by three capital letters (IATA code).
            You may assume all tickets form at least one valid itinerary.
    */

    unordered_map<string, vector<string>> graph;
    for(const auto& t: tickets)
    {
        graph[t[0]].emplace_back(t[1]);
    }

    for(auto& it: graph)
    {
        std::sort(it.second.begin(), it.second.end(), [](const string& l, const string& r) { return l>r;});
    }

    vector<string> ans;
    function<void(string)> dfs = [&](string s)
    {
        ans.push_back(s);

        if(!graph[s].empty())
        {
            auto next = graph[s].back();
            graph[s].pop_back();
            dfs(next);
        }
    };

    dfs("JFK");
    return ans;
}

void findItinerary_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<vector<string>> tickets = stringTo2DArray<string>(input);
    vector<string> actual = ss.findItinerary(tickets);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running findItinerary tests:";
    TIMER_START(findItinerary);
    findItinerary_scaffold("[[MUC, LHR], [JFK, MUC], [SFO, SJC], [LHR, SFO]]", "[JFK, MUC, LHR, SFO, SJC]");
    findItinerary_scaffold("[[JFK,SFO],[JFK,ATL],[SFO,ATL],[ATL,JFK],[ATL,SFO]]", "[JFK,ATL,JFK,SFO,ATL,SFO]");
    TIMER_STOP(findItinerary);
    util::Log(logESSENTIAL) << "findItineray using " << TIMER_MSEC(findItinerary) << " milliseconds";
}
