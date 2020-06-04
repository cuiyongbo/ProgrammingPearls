#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 841, 1202 */

class Solution
{
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms);
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs);
};

bool Solution::canVisitAllRooms(vector<vector<int>>& rooms)
{
    /*
        There are N rooms and you start in room 0.
        Each room has a distinct number in 0, 1, 2, ..., N-1,
        and each room may have some keys to access the next room.

        Formally, each room i has a list of keys rooms[i], and each
        key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.
        A key rooms[i][j] = v opens the room with number v.

        Initially, all the rooms start locked (except for room 0). You can walk back and forth between rooms freely.
        Return true if and only if you can enter every room.
    */

    set<int> unlockedRooms;
    int roomCount = rooms.size();
    function<void(int)> dfs = [&](int room)
    {
        for(const auto& r: rooms[room])
        {
            if(unlockedRooms.count(r) == 0)
            {
                unlockedRooms.emplace(r);
                dfs(r);
            }
        }
    };

    unlockedRooms.emplace(0);

    dfs(0);

    return unlockedRooms.size() == roomCount;
}

string Solution::smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
{
    /*
        You are given a string s, and an array of pairs of indices in the string
        where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.
        You can swap the characters at any pair of indices in the given pairs any number of times.
        Return the lexicographically smallest string that s can be changed to after using the swaps.
    */

    int n = s.size();
    vector<vector<int>> graph(n, vector<int>());
    for(const auto& p: pairs)
    {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }

    string buffer;
    set<int> visited;
    set<int> componet;
    function<void(int)> dfs = [&](int u)
    {
        visited.emplace(u);
        componet.emplace(u);
        buffer.push_back(s[u]);

        for(const auto& v: graph[u])
        {
            if(visited.count(v) == 0)
                dfs(v);
        }
    };

    for(int u=0; u<n; ++u)
    {
        if(visited.count(u) == 0)
        {
            dfs(u);

            int i = 0;
            std::sort(buffer.begin(), buffer.end());
            for(const auto& it: componet)
            {
                s[it] = buffer[i++];
            }

            componet.clear();
            buffer.clear();
        }
    }
    return s;
}

void canVisitAllRooms_scaffold(string input, bool expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input);
    bool actual = ss.canVisitAllRooms(graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << "," << expectedResult  << ") failed";
    }
}

void smallestStringWithSwaps_scaffold(string input, string pairs, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(pairs);
    string actual = ss.smallestStringWithSwaps(input, graph);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << pairs << "," << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << pairs << "," << expectedResult  << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

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
