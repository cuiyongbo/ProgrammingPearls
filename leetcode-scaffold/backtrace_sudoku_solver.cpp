#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 37, 51, 52 */

class Solution {
public:
    void solveSudoku(vector<vector<char>>& board);
    vector<vector<string>> solveNQueens(int n);
    int totalNQueens(int n);
private:
    bool isValidQueen(vector<string>& board, int r, int c);
};

void Solution::solveSudoku(vector<vector<char>>& board) {
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
    auto isValid = [&](int row, int column, int k) {
        for(int i=0; i<len; ++i) {
            if(board[row][i] != '.' && board[row][i]-'0' == k) {
                return false;
            }
            if(board[i][column] != '.' && board[i][column]-'0' == k) {
                return false; 
            }
        }

        // subgrid
        int startRow = row/3*3;
        int startCol = column/3*3;
        for(int i=startRow; i<3+startRow; ++i) {
            for(int j=startCol; j<3+startCol; ++j) {
                if(board[i][j] != '.' && board[i][j]-'0' == k) {
                    return false;
                }
            }
        }
        return true;
    };

    function<bool(int, int)> backtrace = [&] (int r, int c) {
        if (r==len) {
            return true;
        }
        if (c==len) {
            return backtrace(r+1, 0);
        }
        if(board[r][c] != '.') {
            return backtrace(r, c+1);
        } else {
            for(int k=1; k<10; ++k) {
                if(isValid(r, c, k)) {
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

bool Solution::isValidQueen(vector<string>& board, int r, int c) {
    /*
        ↖ ↑ ↗
        ← . →
        ↙ ↓ ↘
    */

    int n = board.size();

    // there is only one queen on each row and column
    for (int i=0; i<n; ++i) {
        if (board[r][i] == 'Q' || board[i][c] == 'Q') {
            return false;
        }
    }

    // there is one queen on each diagonal
    // down right
    for(int i=r, j=c; 0<=i && i<n && 0<=j && j<n; ++i, ++j) {
        if(board[i][j] == 'Q') {
            return false;
        }
    }

    // up left
    for(int i=r, j=c; 0<=i && i<n && 0<=j && j<n; --i, --j) {
        if(board[i][j] == 'Q') {
            return false;
        }
    }

    // left down
    for(int i=r, j=c; 0<=i && i<n && 0<=j && j<n; --i, ++j) {
        if(board[i][j] == 'Q') {
            return false;
        }
    }

    // right up
    for(int i=r, j=c; 0<=i && i<n && 0<=j && j<n; ++i, --j) {
        if(board[i][j] == 'Q') {
            return false;
        }
    }

    return true;
}

vector<vector<string>> Solution::solveNQueens(int n) {
    /*
        The n-queens puzzle is the problem of placing n queens on 
        an n×n chessboard such that no two queens attack each other.

        Given an integer n, return all distinct solutions to the n-queens puzzle.
        Each solution contains a distinct board configuration of the n-queens’ placement, 
        where 'Q' and '.' both indicate a queen and an empty space respectively.
    */

    vector<vector<string>> ans;
    vector<string> board(n, string(n, '.'));
    function<void(int)> backtrace = [&](int r) {
        if (r == n) {
            ans.push_back(board);
            return;
        }
        for (int i=0; i<n; i++) {
            if (isValidQueen(board, r, i)) {
                board[r][i] = 'Q';
                backtrace(r+1);
                board[r][i] = '.';
            }
        }
    };
    backtrace(0);
    return ans;
}

int Solution::totalNQueens(int n) {
    /*
        Given an integer n, return the number of distinct solutions to the n-queens puzzle.
    */

    int ans = 0;
    vector<string> board(n, string(n, '.'));
    function<void(int)> backtrace = [&](int r) {
        if (r == n) {
            ++ans;
            return;
        }
        for (int i=0; i<n; i++) {
            if (isValidQueen(board, r, i)) {
                board[r][i] = 'Q';
                backtrace(r+1);
                board[r][i] = '.';
            }
        }
    };
    backtrace(0);
    return ans;
}

void solveSudoku_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<char>> board = stringTo2DArray<char>(input);
    vector<vector<char>> expected = stringTo2DArray<char>(expectedResult);
    ss.solveSudoku(board);
    if (board == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: board) {
            util::Log(logERROR) << numberVectorToString(s);
        }
    }
}

void solveNQueens_scaffold(int input, string expectedResult) {
    Solution ss;
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> board = ss.solveNQueens(input);
    if(board.size() == expected.size()) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Expected Result size: " << expected.size();
        util::Log(logERROR) << "Actual: " << board.size();

        int id = 0;
        for(const auto& s: board) 
        {
            util::Log(logESSENTIAL) << "Solution " << ++id;
            for(const auto& r: s)
            {
                util::Log(logERROR) << r;
            }
        }
    }
}

void totalNQueens_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.totalNQueens(input);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main() {
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

    util::Log(logESSENTIAL) << "Running solveNQueens tests: ";
    TIMER_START(solveNQueens);

    expectedResult = R"([
        [.Q..,  
        ...Q,
        Q...,
        ..Q.],
        [..Q.,  
         Q...,
         ...Q,
         .Q..]
    ])";

    solveNQueens_scaffold(4, expectedResult);
    solveNQueens_scaffold(1, "[[Q]]");
    solveNQueens_scaffold(2, "[]");
    TIMER_STOP(solveNQueens);
    util::Log(logESSENTIAL) << "solveNQueens using " << TIMER_MSEC(solveNQueens) << " milliseconds"; 

    util::Log(logESSENTIAL) << "Running totalNQueens tests: ";
    TIMER_START(totalNQueens);
    totalNQueens_scaffold(4, 2);
    totalNQueens_scaffold(1, 1);
    totalNQueens_scaffold(2, 0);
    TIMER_STOP(totalNQueens);
    util::Log(logESSENTIAL) << "totalNQueens using " << TIMER_MSEC(totalNQueens) << " milliseconds"; 
}
