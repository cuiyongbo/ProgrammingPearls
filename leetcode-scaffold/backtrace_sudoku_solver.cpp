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


/*
    Write a program to solve a Sudoku puzzle by filling the empty cells. Empty cells are indicated by the character '.'.
    A sudoku solution must satisfy all of the following rules:
        Each of the digits 1-9 must occur exactly once in each row.
        Each of the digits 1-9 must occur exactly once in each column.
        Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
    Note:
        The given board contain only digits 1-9 and the character '.'.
        You may assume that the given Sudoku puzzle will have a single unique solution.
        The given board size is always 9x9.
    chart:       
        row: board[r:]
        column: board[:c]
        subgrid: top-left (r/3*3, c/3*3)  <--- Be cautious!!
        -----------------------
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        -----------------------
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        -----------------------
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        | 1 2 3 | 4 5 6 | 7 8 9 |
        -----------------------
*/
void Solution::solveSudoku(vector<vector<char>>& board) {
    auto is_valid = [&](int r, int c) {
        // row
        for (int i=0; i<9; i++) {
            if (i != c) {
                if (board[r][i] == board[r][c]) {
                    return false;
                }
            }
        }
        // column
        for (int i=0; i<9; i++) {
            if (i != r) {
                if (board[i][c] == board[r][c]) {
                    return false;
                }
            }
        }
        //subgrid
        int sr = r/3*3;
        int sc = c/3*3;
        for (int i=sr; i<sr+3; i++) {
            for (int j=sc; j<sc+3; j++) {
                if (i!=r || j!=c) {
                    if (board[i][j] == board[r][c]) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    function<bool(int, int)> backtrace = [&] (int r, int c) {
        if (r == 9) {
            return true;
        }
        if (c == 9) {
            return backtrace(r+1, 0);
        }
        if (board[r][c] != '.') {
            return backtrace(r, c+1);
        }
        for (int i=1; i<10; i++) {
            board[r][c] = i+'0';
            if (is_valid(r, c)) {
                if (backtrace(r, c+1)) {
                    return true;
                }
            }
            board[r][c] = '.';
        }
        return false;
    };
    backtrace(0, 0);
    return;
}


bool Solution::isValidQueen(vector<string>& board, int r, int c) {
    int n = board.size();
    for (int i=0; i<n; ++i) {
        if (i!=c && board[r][i]=='Q') { // not necessary
            return false;
        }
        if (i!=r && board[i][c]=='Q') { // column
            return false;
        }
    }
    // check uniqueness alone each diagonal
    // we take (r, c) as one end of each of the four diagonals: upper left, down right, down left, uppper right
    int i=r, j=c;
    while (i>=0 && j>=0) { // upper left
        if (board[i][j] == 'Q') {
            return false;
        }
        i--; j--;
    }
    i=r, j=c;
    while (i<n && j<n) { // down right
        if (board[i][j] == 'Q') {
            return false;
        }
        i++; j++;
    }
    i=r, j=c;
    while (i<n && j>=0) { // down left
        if (board[i][j] == 'Q') {
            return false;
        }
        i++; j--;
    }
    i=r, j=c;
    while (i>=0 && j<n) { // upper right
        if (board[i][j] == 'Q') {
            return false;
        }
        i--; j++;
    }
    return true;
}


/*
    The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
    Given an integer n, return all distinct solutions to the n-queens puzzle.
    Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' indicate a queen and an empty space respectively.
    A valid configuration is such one that none of the n Queens share the same row, column, or diagonal. In this case, "diagonal" means all diagonals, not just the two that bisect the board.
*/
vector<vector<string>> Solution::solveNQueens(int n) {
    vector<vector<string>> ans;
    vector<string> board(n, string(n, '.'));
    function<void(int r)> backtrace = [&](int r) {
        if (r == n) {
            ans.push_back(board);
            return;
        }
        for (int c=0; c<n; c++) {
            // prune invalid branches
            if (isValidQueen(board, r, c)) {
                board[r][c] = 'Q';
                backtrace(r+1);
                board[r][c] = '.';
            }
        }
    };
    backtrace(0);
    return ans;
}


/*
    Given an integer n, return the number of distinct solutions to the n-queens puzzle.
*/
int Solution::totalNQueens(int n) {
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
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual:", input, expectedResult);
        for (const auto& s: board) {
            SPDLOG_ERROR(numberVectorToString(s));
        }
    }
}


void solveNQueens_scaffold(int input, string expectedResult) {
    Solution ss;
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> board = ss.solveNQueens(input);
    if (board.size() == expected.size()) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, expected Result size: {}, actual: {}", input, expectedResult, expected.size(), board.size());
        int id = 0;
        for (const auto& s: board) {
            std::cout << "Solution " << ++id << endl;
            for (const auto& r: s) {
                std::cout << r << endl;
            }
        }
    }
}


void totalNQueens_scaffold(int input, int expectedResult) {
    Solution ss;
    int actual = ss.totalNQueens(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed, actual: {}", input, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running solveSudoku tests:");
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
    SPDLOG_WARN("solveSudoku using {} ms", TIMER_MSEC(solveSudoku));

    SPDLOG_WARN("Running solveNQueens tests:");
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
    solveNQueens_scaffold(3, "[]");
    TIMER_STOP(solveNQueens);
    SPDLOG_WARN("solveNQueens using {} ms", TIMER_MSEC(solveNQueens));

    SPDLOG_WARN("Running totalNQueens tests:");
    TIMER_START(totalNQueens);
    totalNQueens_scaffold(4, 2);
    totalNQueens_scaffold(1, 1);
    totalNQueens_scaffold(2, 0);
    totalNQueens_scaffold(3, 0);
    TIMER_STOP(totalNQueens);
    SPDLOG_WARN("totalNQueens using {} ms", TIMER_MSEC(totalNQueens));
}
