#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 443 */

class Solution
{
public:
    int minMutation(string start, string end, vector<string>& bank);
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

            visited.emplace(s);

            for(const auto& b: bank)
            {
                if(visited.count(b) != 0) continue;
                if(!isValidMutation(s, b)) continue;
                q.push(b);
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

}
