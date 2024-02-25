#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 79, 212 */

class Solution {
public:
    bool exist(const vector<vector<char>>& board, const string& word);
    vector<string> findWords(const vector<vector<char>>& board, const vector<string>& words);
};


/*
    Given a 2D board and a word, find if the word exists in the grid.
    The word can be constructed from letters of sequentially adjacent cell, 
    where “adjacent” cells are those horizontally or vertically neighboring. 
    **The same letter cell may not be used more than once.**
*/
bool Solution::exist(const vector<vector<char>>& board, const string& word) {
    int rows = board.size();
    int columns = board[0].size();
    vector<vector<bool>> used(rows, vector<bool>(columns, false));
    // take care of corner case: the length of word is equal to rows*columns
    function<bool(int, int, int)> backtrace = [&] (int r, int c, int u) {
        if (u == word.size()) {
            return true;
        }
        if (r<0 || r>=rows || c<0 || c>=columns || used[r][c]) {
            return false;
        }
        if (board[r][c] != word[u]) {
            return false;
        }
        used[r][c] = true;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (backtrace(nr, nc, u+1)) {
                return true;
            }
        }
        used[r][c] = false;
        return false;
    };
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<columns; ++c) {
            if (backtrace(r, c, 0)) {
                return true;
            }
        }
    }
    return false;
}


/*
    Given a 2D board and a list of words from the dictionary, find all words in the board.
    Each word must be constructed from letters of sequentially adjacent cell, 
    where “adjacent” cells are those horizontally or vertically neighboring. 
    **The same letter cell may not be used more than once in a word.**
*/
vector<string> Solution::findWords(const vector<vector<char>>& board, const vector<string>& words) {
    vector<string> ans;
    for (auto& s: words) {
        if (exist(board, s)) {
            ans.push_back(s);
        }
    }
    return ans;
}


void exist_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input1);
    bool actual = ss.exist(board, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}


void findWords_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input1);
    vector<string> words = stringTo1DArray<string>(input2);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.findWords(board, words);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running exist tests: ";
    TIMER_START(exist);

    string board = R"([
        [A,B,C,E],
        [S,F,C,S],
        [A,D,E,E]
    ])";

    exist_scaffold(board, "ABCCED", true);
    exist_scaffold(board, "SEE", true);
    exist_scaffold(board, "ABCB", false);
    exist_scaffold(board, "SFCSA", false);
    exist_scaffold(board, "ABCES", true);
    exist_scaffold(board, "SADECS", true);
    exist_scaffold(board, "CESCC", false);
    exist_scaffold(board, "EEE", false);
    exist_scaffold("[[E]]", "E", true);
    exist_scaffold("[[H,E,L,L,O]]", "HELLO", true);

    TIMER_STOP(exist);
    util::Log(logESSENTIAL) << "exist using " << TIMER_MSEC(exist) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running findWords tests: ";
    TIMER_START(findWords);

    board = R"([
      [o,a,a,n],
      [e,t,a,e],
      [i,h,k,r],
      [i,f,l,v]
    ])";

    findWords_scaffold(board, "[oath,pea,eat,rain]", "[eat,oath]");

    TIMER_STOP(findWords);
    util::Log(logESSENTIAL) << "findWords using " << TIMER_MSEC(findWords) << " milliseconds"; 
}
