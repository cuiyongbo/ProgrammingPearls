#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 208 */

void TrieTree_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    TrieTree tm;
    int n = (int)funcOperations.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "insert")
        {
            tm.insert(funcArgs[i][0]);
        }
        else if(funcOperations[i] == "search" || funcOperations[i] == "startsWith")
        {
            bool actual = funcOperations[i] == "search" ? 
                                tm.search(funcArgs[i][0]) :
                                tm.startsWith(funcArgs[i][0]);

            string actual_str = actual ? "true" : "false";
            if(actual_str != ans[i])
            {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
            else
            {
                util::Log(logESSENTIAL) << funcOperations[i] << "(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running TrieTree tests:";
    TIMER_START(TrieTree);
    TrieTree_scaffold(
        "[TrieTree,insert,insert,insert,search,search,startsWith]", 
        "[[],[hello],[heros],[hell],[hello],[hero],[hero]]",
        "[null,null,null,null,true,false,true]");
    TIMER_STOP(TrieTree);
    util::Log(logESSENTIAL) << "TrieTree using " << TIMER_MSEC(TrieTree) << " milliseconds";
}
