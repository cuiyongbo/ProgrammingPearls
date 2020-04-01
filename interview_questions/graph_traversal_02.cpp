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
        You are given a string s, and an array of pairs of indices in the string pairs
        where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.
        You can swap the characters at any pair of indices in the given pairs any number of times.
        Return the lexicographically smallest string that s can be changed to after using the swaps.
    */



}

void canVisitAllRooms_scaffold(string input, string pairs, string expectedResult)
{
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray(pairs);
    string actual = ss.canVisitAllRooms(graph);
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
    canVisitAllRooms_scaffold("dcab", "[[0,3],[1,2]]", "bacd");
    canVisitAllRooms_scaffold("dcab", "[[0,3],[1,2],[0,2]]", "abcd");
    canVisitAllRooms_scaffold("cba", "[[0,1],[1,2]]", "abc");
    TIMER_STOP(smallestStringWithSwaps);
    util::Log(logESSENTIAL) << "smallestStringWithSwaps using " << TIMER_MSEC(smallestStringWithSwaps) << " milliseconds";
}
