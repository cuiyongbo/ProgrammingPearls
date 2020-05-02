#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 79, 212 */

class Solution 
{
public:
    bool exist(const vector<vector<char>> &board, const string& word);
    vector<string> findWords(const vector<vector<char>>& board, const vector<string>& words);
};

bool Solution::exist(const vector<vector<char>>& board, const string& word)
{
    /*
        Given a 2D board and a word, find if the word exists in the grid.

        The word can be constructed from letters of sequentially adjacent cell, 
        where “adjacent” cells are those horizontally or vertically neighboring. 
        The same letter cell may not be used more than once.
    */

    int rows = (int)board.size();
    int columns = (int)board[0].size();

    string courier;
    vector<vector<bool>> used(rows, vector<bool>(columns, false));
    auto isValid = [&](int r, int c)
    {
        return courier.size() < word.size() && 
                0<=r && r<rows && 
                0<=c && c<columns && 
                !used[r][c] &&
                board[r][c] == word[courier.size()];
    };

    function<bool(int,int)> backtrace = [&](int i, int j)
    {
        if(courier == word) return true;
        for(const auto& d: DIRECTIONS)
        {
            int r = i+d[1];
            int c = j+d[0];
            if(isValid(r, c))
            {
                used[r][c] = true;
                courier.push_back(board[r][c]);
                if(backtrace(r, c)) return true;
                courier.pop_back();
                used[r][c] = false;
            }
        }
        return false;
    };

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            if(backtrace(i, j)) 
                return true;
        }
    }
    return false;
}

vector<string> Solution::findWords(const vector<vector<char>>& board, const vector<string>& words)
{
    /*
        Given a 2D board and a list of words from the dictionary, find all words in the board.

        Each word must be constructed from letters of sequentially adjacent cell, 
        where “adjacent” cells are those horizontally or vertically neighboring. 
        The same letter cell may not be used more than once in a word.
    */

    vector<string> ans;
    for(const auto& s: words)
    {
        if(exist(board, s))
            ans.push_back(s);
    }

    // not necessary, out of convenience for test
    std::sort(ans.begin(), ans.end());
    return ans;
}

void exist_scaffold(string input1, string input2, bool expectedResult)
{
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input1);
    bool actual = ss.exist(board, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void findWords_scaffold(string input1, string input2, string expectedResult)
{
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input1);
    vector<string> words = stringTo1DArray<string>(input2);
    vector<string> expected = stringTo1DArray<string>(expectedResult);
    vector<string> actual = ss.findWords(board, words);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << s;
    }
}

int main()
{
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
