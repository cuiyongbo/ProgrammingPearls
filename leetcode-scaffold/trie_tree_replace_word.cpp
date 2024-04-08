#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode: 648, 720 */

class Solution {
public:
    string replaceWords(vector<string>& dict, string sentence);
    string longestWord(vector<string>& words);
};

/*
    In English, we have a concept called root, which can be followed by some other words to form another longer word – let’s call this word successor. 
    For example, the root `an`, followed by `other`, which can form another word `another`.
    Now, given a dictionary consisting of many roots and a sentence. You need to replace all the successor in the sentence with the root forming it. 
    If a successor has many roots can form it, replace it with the root with the shortest length. You need to output the sentence after the replacement.
    You may assume that all the inputs consist of lowercase letters a-z.
*/
string Solution::replaceWords(vector<string>& dict, string sentence) {
    TrieTree tree;
    for (auto& d: dict) {
        tree.insert(d);
    }
    auto word_root = [&] (string word) -> string {
        TrieNode* p = tree.root();
        string root;
        for (auto c: word) {
            if (p->children[c] == nullptr) {
                break;
            }
            root.push_back(c);
            p = p->children[c];
            if (p->is_leaf) {
                return root;
            }
        }
        return word;
    };
    string ans;
    // split sentence by whitespace
    std::stringstream ss(sentence);
    for (std::string word; std::getline(ss, word, ' ');) {
        ans.append(word_root(word) + " ");
    }
    ans.pop_back();
    return ans;
}

/*
    Given a list of strings words representing an English Dictionary, find the longest word in words that can be built one character at 
    a time by other words in words. If there is more than one possible answer, return the longest word with the smallest lexicographical order.
    If there is no answer, return the empty string. For example, given an input words=[w,wo,wor,worl,world], the output is "world",
    The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
*/
string Solution::longestWord(vector<string>& words) {
    TrieTree tree;
    for (const auto& w: words) {
        tree.insert(w);
    }
    string ans;
    string buffer;
    function<void(TrieNode*)> backtrace = [&] (TrieNode* p) {
        if (ans.size() < buffer.size()) {
            ans = buffer;
        }
        for (int i='a'; i<='z'; ++i) {
            auto ch = p->children[i];
            // we expect a leaf node at every step
            if (ch == nullptr || !ch->is_leaf) {
                continue;
            }
            buffer.push_back(i);
            backtrace(ch);
            buffer.pop_back();
        }
    };
    backtrace(tree.root());
    return ans;
}

void replaceWords_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input1);
    string actual = ss.replaceWords(dict, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void longestWord_scaffold(string input1, string expectedResult) {
    Solution ss;
    vector<string> dict = stringTo1DArray<string>(input1);
    string actual = ss.longestWord(dict);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running replaceWords tests:";
    TIMER_START(replaceWords);
    replaceWords_scaffold("[cat,bat,rat]", "the cattle was rattled by the battery", "the cat was rat by the bat");
    TIMER_STOP(replaceWords);
    util::Log(logESSENTIAL) << "replaceWords using " << TIMER_MSEC(replaceWords) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestWord tests:";
    TIMER_START(longestWord);
    longestWord_scaffold("[w,wo,wor,worl,world]", "world");
    longestWord_scaffold("[a, banana, app, appl, ap, apply, apple]", "apple");
    longestWord_scaffold("[yo,ew,fc,zrc,yodn,fcm,qm,qmo,fcmz,z,ewq,yod,ewqz,y]", "yodn");
    longestWord_scaffold("[cat,banana,dog,nana,walk,walker,dogwalker]", "");
    TIMER_STOP(longestWord);
    util::Log(logESSENTIAL) << "longestWord using " << TIMER_MSEC(longestWord) << " milliseconds";

}
