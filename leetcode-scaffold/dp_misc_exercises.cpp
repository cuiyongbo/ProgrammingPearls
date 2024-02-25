#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 1139, 688, 576, 935, 322, 377  */

class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid);
    double knightProbability(int n, int k, int row, int column);
    int findPaths(int m, int n, int maxMove, int startRow, int startColumn);
    int knightDialer(int n);
    int coinChange(vector<int>& coins, int amount);
    int combinationSum_322(vector<int>& nums, int target);


};

int Solution::largest1BorderedSquare(vector<vector<int>>& grid) {
/*
    Given a 2D grid of 0s and 1s, return the number of elements in the largest *square* subgrid that has all 1s on its border, or 0 if such a subgrid doesn't exist in the grid.
*/


    return 0;
}

double knightProbability(int n, int k, int row, int column) {
/*
On an n x n chessboard, a knight starts at the cell (row, column) and attempts to make exactly k moves. The rows and columns are 0-indexed, so the top-left cell is (0, 0), and the bottom-right cell is (n - 1, n - 1).
A chess knight has eight possible moves it can make, as illustrated below. Each move is two cells in a cardinal direction, then one cell in an orthogonal direction.
Each time the knight is to move, it chooses one of eight possible moves uniformly at random (even if the piece would go off the chessboard) and moves there.
The knight continues moving until it has made exactly k moves or has moved off the chessboard.
Return the probability that the knight remains on the board after it has stopped moving.
*/
    return 0;
}

int Solution::findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
/*
    There is an m x n grid with a ball. The ball is initially at the position [startRow, startColumn]. You are allowed to move the ball to one of the four adjacent cells in the grid (possibly out of the grid crossing the grid boundary). You can apply at most maxMove moves to the ball.

    Given the five integers m, n, maxMove, startRow, startColumn, return the number of paths to move the ball out of the grid boundary. Since the answer can be very large, return it modulo 10^9 + 7.
*/
    const int mod = 10^9 + 7;
    function<int(int, int, int)> dfs = [&] (int r, int c, int move) {
        if (move >= maxMove) {
            return 0;
        }
        int count = 0;
        for (auto& d: directions) {
            int nr = r + d.first;
            int nc = c + d.second;
            if (nr<0 || nr>=m || nc<0 || nc>=n) {
                count = (count+1)%mod;
                continue;
            }
            count = (count + dfs(nr, nc, move+1))%mod;
        }
        return count;
    };
    return dfs(startRow, startColumn, 0);
}

int Solution::knightDialer(int n) {
/*
    Given an integer n, return how many distinct phone numbers of length n we can dial.
    You are allowed to place the knight on any numeric cell initially and then you should perform n - 1 jumps to dial a number of length n.
    All jumps should be valid knight jumps. As the answer may be very large, return the answer modulo 109 + 7.
*/
    return 0;
}

int Solution::coinChange(vector<int>& coins, int amount) {
/*
    You are given coins of different denominations and a total amount of money amount.
    Write a function to compute the fewest number of coins that you need to make up that amount.
    If that amount of money cannot be made up by any combination of the coins, return -1.
    Example 1:
        coins = [1, 2, 5], amount = 11
        return 3 (11 = 5 + 5 + 1)
*/
    // dp[i] means min number of coins to make up amount i
    vector<int> dp(amount+1, INT32_MAX); dp[0] = 0;
    for (int coin: coins) {
        for (int i=coin; i<=amount; ++i) {
            if (dp[i-coin] != INT32_MAX) {
                dp[i] = min(dp[i], dp[i-coin]+1);
            }
        }
    }
    return dp[amount]==INT32_MAX ? -1 : dp[amount];
}

int Solution::combinationSum_322(vector<int>& nums, int target) {
/*
    Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.
    Given inputs: nums = [1, 2, 3], target = 4, The possible combination ways are:
        (1, 1, 1, 1)
        (1, 1, 2)
        (1, 2, 1)
        (1, 3)
        (2, 1, 1)
        (2, 2)
        (3, 1)
    Note that different sequences are counted as different combinations. Therefore the output is 7.
*/
    // dp[i] means the number of combination whose sum up to i
    vector<int> dp(target+1, 0); dp[0] = 1;
    for (int i=1; i<=target; ++i) {
        for (int n: nums) {
            if (i-n >= 0) {
                dp[i] += dp[i-n];
            }
        }
    }
    return dp[target];
}


void largest1BorderedSquare_scaffold(string input, int expectedResult) {
    Solution ss;
    auto grid = stringTo2DArray<int>(input);
    int actual = ss.largest1BorderedSquare(grid);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void findPaths_scaffold(string input, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input);
    int actual = ss.findPaths(vi[0], vi[1], vi[2], vi[3], vi[4]);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void coinChange_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input1);
    int actual = ss.coinChange(vi, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void combinationSum_322_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto vi = stringTo1DArray<int>(input1);
    int actual = ss.combinationSum_322(vi, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running largest1BorderedSquare tests:";
    TIMER_START(largest1BorderedSquare);
    largest1BorderedSquare_scaffold("[[1,1,1],[1,0,1],[1,1,1]]", 9);
    largest1BorderedSquare_scaffold("[[1,1],[0,0]]", 1);
    TIMER_STOP(largest1BorderedSquare);
    util::Log(logESSENTIAL) << "largest1BorderedSquare using " << TIMER_MSEC(largest1BorderedSquare) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findPaths tests:";
    TIMER_START(findPaths);
    findPaths_scaffold("[2,2,2,0,0]", 6);
    findPaths_scaffold("[1,3,3,0,1]", 12);
    findPaths_scaffold("[1,2,50,0,0]", 150);
    TIMER_STOP(findPaths);
    util::Log(logESSENTIAL) << "findPaths using " << TIMER_MSEC(findPaths) << " milliseconds";

    util::Log(logESSENTIAL) << "Running coinChange tests:";
    TIMER_START(coinChange);
    coinChange_scaffold("[1,2,5]", 11, 3);
    coinChange_scaffold("[2]", 3, -1);
    TIMER_STOP(coinChange);
    util::Log(logESSENTIAL) << "coinChange using " << TIMER_MSEC(coinChange) << " milliseconds";

    util::Log(logESSENTIAL) << "Running combinationSum_322 tests:";
    TIMER_START(combinationSum_322);
    combinationSum_322_scaffold("[1,2,3]", 4, 7);
    combinationSum_322_scaffold("[9]", 4, 0);
    TIMER_STOP(combinationSum_322);
    util::Log(logESSENTIAL) << "combinationSum_322 using " << TIMER_MSEC(combinationSum_322) << " milliseconds";

}
