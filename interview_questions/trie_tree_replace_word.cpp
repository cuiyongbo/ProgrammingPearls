#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 648 */

class Solution 
{
public:
    string replaceWords(vector<string>& dict, string sentence);
};

string Solution::replaceWords(vector<string>& dict, string sentence)
{
    /*
        In English, we have a concept called root, which can be followed by 
        some other words to form another longer word – let’s call this word successor. 
        For example, the root an, followed by other, which can form another word another.

        Now, given a dictionary consisting of many roots and a sentence. 
        You need to replace all the successor in the sentence with the root forming it. 
        If a successor has many roots can form it, replace it with the root with the shortest length.

        You need to output the sentence after the replacement.

        You may assume that all the inputs are consist of lowercase letters a-z.
    */

    TrieTree tt;
    for(const auto& s: dict) tt.insert(s);
    auto searchRoot = [&](const string& word)
    {
        string ans;
        TrieNode* current = tt.root();
        for(const auto& c: word)
        {
            current = current->children[c-'a'];
            if(current == NULL) break;
            ans.push_back(c);
            if(current->is_leaf) break;
        }
        return (current != NULL && current->is_leaf) ? ans : word;
    };

    string ans;
    stringstream ss(sentence);
    string item;
    const char delimiter = ' ';
    while(std::getline(ss, item, delimiter))
    {
        string r = searchRoot(item);
        if(!ans.empty()) ans.push_back(' ');
        ans.append(r);
    }
    return ans;
}

void replaceWords_scaffold(string input1, string input2, string expectedOutputs)
{
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input1);
    string actual = ss.replaceWords(dict, input2);
    if(actual == expectedOutputs)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedOutputs: " << expectedOutputs << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedOutputs: " << expectedOutputs << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running replaceWords tests:";
    TIMER_START(replaceWords);
    replaceWords_scaffold("[cat,bat,rat]", "the cattle was rattled by the battery", "the cat was rat by the bat");
    TIMER_STOP(replaceWords);
    util::Log(logESSENTIAL) << "replaceWords using " << TIMER_MSEC(replaceWords) << " milliseconds";


}
