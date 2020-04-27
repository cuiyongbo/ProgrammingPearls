#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 37 */

class Solution
{
public:
    void solveSudoku(vector<vector<char>>& board);

};

void Solution::solveSudoku(vector<vector<char>>& board)
{
    /*
        Write a program to solve a Sudoku puzzle by filling the empty cells.
        Empty cells are indicated by the character '.'.

        A sudoku solution must satisfy all of the following rules:

            Each of the digits 1-9 must occur exactly once in each row.
            Each of the digits 1-9 must occur exactly once in each column.
            Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
        
        Note:

            The given board contain only digits 1-9 and the character '.'.
            You may assume that the given Sudoku puzzle will have a single unique solution.
            The given board size is always 9x9.
    */

    const int len = 9;
    auto isValid = [&](int row, int column, int k)
    {
        for(int i=0; i<len; ++i)
        {
            if(board[row][i] != '.' && board[row][i]-'0' == k)
                return false;
            if(board[i][column] != '.' && board[i][column]-'0' == k)
                return false; 
        }

        // sub grid
        int startRow = row/3*3;
        int startCol = column/3*3;
        for(int i=startRow; i<3+startRow; ++i)
        {
            for(int j=startCol; j<3+startCol; ++j)
            {
                if(board[i][j] != '.' && board[i][j]-'0' == k)
                    return false;
            }
        }
        return true;
    };

    function<bool(int, int)> backtrace = [&] (int r, int c)
    {
        if(r==len) return true;
        if(c==len) return backtrace(r+1, 0);
        if(board[r][c] != '.')
        {
            return backtrace(r, c+1);
        }
        else
        {
            for(int k=1; k<10; ++k)
            {
                if(isValid(r, c, k))
                {
                    board[r][c] = '0' + k;
                    if(backtrace(r, c+1))
                        return true;
                    board[r][c] = '.';
                }
            }
            return false;
        }
    };

    backtrace(0, 0);

#ifdef DEBUG
util::Log(logESSENTIAL) << "Debug...";
for(int i=0; i<len; i++)
{
    for(int j=0; j<len; j++)
    {
        if(!isValid(i, j))
        {
            util::Log(logERROR) << "solveSudoku failed: (" << i << ", " << j << ")";
            return;
        }
    }
}
#endif

}

void solveSudoku_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input);
    vector<vector<char>> expected = stringTo2DArray<char>(expectedResult);
    ss.solveSudoku(board);
    if(board == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: board) util::Log(logERROR) << numberVectorToString(s);
    }
}


int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running solveSudoku tests: ";
    TIMER_START(solveSudoku);

    string board = R"([
        [5,3,.,.,7,.,.,.,.],
        [6,.,.,1,9,5,.,.,.],
        [.,9,8,.,.,.,.,6,.],
        [8,.,.,.,6,.,.,.,3],
        [4,.,.,8,.,3,.,.,1],
        [7,.,.,.,2,.,.,.,6],
        [.,6,.,.,.,.,2,8,.],
        [.,.,.,4,1,9,.,.,5],
        [.,.,.,.,8,.,.,7,9]])";

    string expectedResult = R"([
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9]])";

    solveSudoku_scaffold(board, expectedResult);
    TIMER_STOP(solveSudoku);
    util::Log(logESSENTIAL) << "solveSudoku using " << TIMER_MSEC(solveSudoku) << " milliseconds"; 

}
