#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 211 */

/*
Design a data structure that supports the following two operations:

    void addWord(word)
    bool search(word)

search(word) can search a literal word or a regular expression string containing only 
letters a-z or .. A . means it can represent any one letter.

Example:

    addWord("bad")
    addWord("dad")
    addWord("mad")
    search("pad") -> false
    search("bad") -> true
    search(".ad") -> true
    search("b..") -> true

Note: You may assume that all words are consist of lowercase letters a-z.
*/

class WordDictionary 
{
public:
    void addWord(const string& word);
    bool search(const string& word);

private:
    TrieTree m_tree;
};

void WordDictionary::addWord(const string& word)
{
    m_tree.insert(word);
}
    
/** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
bool WordDictionary::search(const string& word)
{
    size_t word_len = word.size();
    function<bool(TrieNode*, int)> dfs = [&](TrieNode* cur, size_t p)
    {
        if(cur == NULL) return false;
        if(p == word_len) return cur->is_leaf;

        if(word[p] != '.')
        {
            cur = cur->children[word[p] - 'a'];
            return dfs(cur, p+1);
        }
        else
        {
            for(auto c: cur->children)
            {
                if(c != NULL)
                {
                    if(dfs(c, p+1)) 
                        return true;
                }
            }
            return false;
        }
    };
    return dfs(m_tree.root(), 0);
}

void WordFilter_scaffold(string operations, string args, string expectedOutputs)
{
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    WordDictionary tm;
    int n = (int)funcOperations.size();
    for(int i=0; i<n; ++i)
    {
        if(funcOperations[i] == "addWord")
        {
            tm.addWord(funcArgs[i][0]);
        }
        else if(funcOperations[i] == "search")
        {
            bool actual = tm.search(funcArgs[i][0]);
            string actual_str = actual ? "true" : "false";
            if(actual_str != ans[i])
            {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual_str;
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
    
    util::Log(logESSENTIAL) << "Running WordDictionary tests:";
    TIMER_START(WordDictionary);
    WordFilter_scaffold(
        "[WordDictionary,addWord,addWord,addWord,search,search,search,search,search]", 
        "[[],[bad],[dad],[mad],[pad],[bad],[.ad],[b..],[...]]",
        "[null,null,null,null,false,true,true,true,true]");
    TIMER_STOP(WordDictionary);
    util::Log(logESSENTIAL) << "WordDictionary using " << TIMER_MSEC(WordDictionary) << " milliseconds";
}
