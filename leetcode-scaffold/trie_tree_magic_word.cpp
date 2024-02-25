#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode: 676 */

/*
    Implement a magic directory with buildDict, and search methods.
    For the method buildDict, you’ll be given a list of non-repetitive words to build a dictionary.
    For the method search, you’ll be given a word, and judge whether if you modify exactly one character 
    into another character in this word, the modified word is in the dictionary you just built.
    Note: You may assume that all the inputs are consist of lowercase letters a-z.

    Example 1:
        Input: buildDict(["hello", "leetcode"]), Output: Null
        Input: search("hello"), Output: False
        Input: search("hhllo"), Output: True
        Input: search("hell"), Output: False
        Input: search("leetcoded"), Output: False
*/

class MagicDictionary {
public:
    void buildDict(const vector<string>& dict);
    bool search(string word);
private:
    TrieTree m_trieTree;
};

void MagicDictionary::buildDict(const vector<string>& dict) {
    for (const auto& s: dict) {
        m_trieTree.insert(s);
    }
}

bool MagicDictionary::search(string word) {

{ // naive method
    for (int i=0; i<word.size(); ++i) {
        for (char c='a'; c<='z'; ++c) {
            if (word[i] == c) {
                continue;
            }
            char prev = word[i];
            word[i] = c;
            if (m_trieTree.search(word)) {
                return true;
            }
            word[i] = prev;
        }
    }   
    return false;
}

{
    string candidate;
    int word_sz = word.size();
    auto is_valid = [&] (string input) {
        if (input.size() != word_sz) {
            return false;
        }
        int diff = 0;
        for (int i=0; i<word_sz && diff<2; ++i) {
            if (input[i] != word[i]) {
                ++diff;
            }
        }
        return diff == 1;
    };
    function<bool(TrieNode*)> backtrace = [&] (TrieNode* cur) {
        if (cur == nullptr || candidate.size() > word_sz) {
            return false;
        }
        if (cur->is_leaf) {
            if (is_valid(candidate)) {
                return true;
            }
        }
        for (int i=0; i<cur->children.size(); ++i) {
            candidate.push_back(i+'a');
            if (backtrace(cur->children[i])) {
                return true;
            }
            candidate.pop_back();
        }
        return false;
    };
    return backtrace(m_trieTree.root());
}

}

void MagicDictionary_scaffold(string operations, string args, string expectedOutputs) {
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(args);
    vector<string> ans = stringTo1DArray<string>(expectedOutputs);
    MagicDictionary tm;
    int n = funcOperations.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "buildDict") {
            tm.buildDict(funcArgs[i]);
        } else if (funcOperations[i] == "search") {
            bool actual = tm.search(funcArgs[i][0]);
            string actual_str = actual ? "true" : "false";
            if (actual_str != ans[i]) {
                util::Log(logERROR) << funcOperations[i] << "(" << funcArgs[i][0] << ") failed. Expected: " << ans[i] << ", actual: " << actual;
            } else {
                util::Log(logINFO) << funcOperations[i] << "(" << funcArgs[i][0] << ") passed";
            }
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    
    util::Log(logESSENTIAL) << "Running MagicDictionary tests:";
    TIMER_START(MagicDictionary);
    MagicDictionary_scaffold(
        "[MagicDictionary,buildDict,search,search,search,search,search,search]", 
        "[[],[hello,leetcode,hero],[hello],[hhllo],[hell],[leetcoded],[hellp],[pello]]",
        "[null,null,false,true,false,false,true,true]");
    TIMER_STOP(MagicDictionary);
    util::Log(logESSENTIAL) << "MagicDictionary using " << TIMER_MSEC(MagicDictionary) << " milliseconds";
}
