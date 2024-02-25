#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 303, 304 */

class NumArray {
/*
    leetcode: 304
    Given an integer array nums, find the sum of the elements between indices i and j (i≤j), inclusive.
    Example: Given nums = [-2, 0, 3, -5, 2, -1]
        sumRange(0, 2) -> 1
        sumRange(2, 5) -> -1
        sumRange(0, 5) -> -3
    Note:
        You may assume that the array does not change. There are many calls to sumRange function.
*/
public:
    NumArray(const vector<int>& nums) {
        // m_sum[i] means sum(nums[0:i]), i is not inclusive
        m_sum.resize(nums.size() + 1, 0);
        std::partial_sum(nums.begin(), nums.end(), std::next(m_sum.begin()));
    }

    int sumRange(int i, int j) { return m_sum[j+1] - m_sum[i];}

private:
    vector<int> m_sum;
};

void NumArray_scaffold(string input1, string input2, string input3, string input4) {
    vector<string> funcs = stringTo1DArray<string> (input1);
    vector<vector<string>> funcArgs = stringTo2DArray<string>(input2);
    vector<int> arr = stringTo1DArray<int> (input3);
    vector<int> expected = stringTo1DArray<int> (input4);
    int sz = funcs.size();
    NumArray ranger(arr);
    for (int i=0; i<sz; ++i) {
        if (funcs[i] == "sumRange") {
            int actual = ranger.sumRange(std::stoi(funcArgs[i][0]), std::stoi(funcArgs[i][1]));
            if (actual == expected[i]) {
                util::Log(logINFO) << funcs[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") test passed";
            } else {
                util::Log(logERROR) << funcs[i] << "(" << funcArgs[i][0] << "," << funcArgs[i][1] << ") test failed, "
                            << "actual: " << actual << ", expected: " << expected[i];
            }
        }
    }
}

class NumMatrix { 
/*
    leetcode 304: 
    Given a 2D grid grid, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
    (row1, col1), (row2, col2) are 0-indexed.
*/
public:
    NumMatrix(vector<vector<int>>& grid);
    int sumRegion(int row1, int col1, int row2, int col2);
private:
    // m_dp[i][j] means the sum of matrix whose upper-left resides in (0, 0) and bottom-right in (i-1, j-1)
    vector<vector<int>> m_dp;
};

NumMatrix::NumMatrix(vector<vector<int>>& grid) {
    int rows = grid.size();
    int columns = grid[0].size();
    // dp[i][j] means sum([0:i, 0:j]), not inclusive
    vector<vector<int>> dp(rows+1, vector<int>(columns+1, 0));
    for (int r=1; r<=rows; ++r) {
        for (int c=1; c<=columns; ++c) { // sum by row
            dp[r][c] += dp[r][c-1];
            dp[r][c] += grid[r-1][c-1];
        }
    }
    for (int c=1; c<=columns; ++c) {
        for (int r=1; r<=rows; ++r) { // sum by column
            dp[r][c] += dp[r-1][c];
        }
    }
    m_dp.swap(dp);

#ifdef DEBUG
    cout << "grid: " << endl;
    for (auto& v: grid) {
        cout << numberVectorToString(v) << endl;
    }
    cout << "m_dp: " << endl;
    for (auto& v: m_dp) {
        cout << numberVectorToString(v) << endl;
    }
#endif
}

int NumMatrix::sumRegion(int row1, int col1, int row2, int col2) {
    return m_dp[row2+1][col2+1] - m_dp[row2+1][col1] - m_dp[row1][col2+1] + m_dp[row1][col1];
}

void NumMatrix_scaffold(string input, string operations, string args, string expectedOutputs) {
    vector<vector<int>> grid = stringTo2DArray<int>(input);
    vector<string> funcOperations = stringTo1DArray<string>(operations);
    vector<vector<int>> funcArgs = stringTo2DArray<int>(args);
    vector<int> ans = stringTo1DArray<int>(expectedOutputs);
    NumMatrix tm(grid);
    int n = ans.size();
    for (int i=0; i<n; ++i) {
        if (funcOperations[i] == "sumRegion") {
            int actual = tm.sumRegion(funcArgs[i][0], funcArgs[i][1], funcArgs[i][2], funcArgs[i][3]);
            if (actual == ans[i]) {
                util::Log(logINFO) << funcOperations[i] << "(" << numberVectorToString<int>(funcArgs[i]) << ") passed";
            } else {
                util::Log(logERROR) << funcOperations[i] << "(" << numberVectorToString<int>(funcArgs[i]) << ") failed";
                util::Log(logERROR) << "Expected: " << ans[i] << ", actual: " << actual;
            }
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running NumArray tests:";
    TIMER_START(NumArray);
    NumArray_scaffold("[NumArray,sumRange,sumRange,sumRange,sumRange]", 
                    "[[],[0,2],[2,5],[0,5],[0,0]]",
                    "[-2, 0, 3, -5, 2, -1]",
                    "[0,1,-1,-3,-2]");
    TIMER_STOP(NumArray);
    util::Log(logESSENTIAL) << "NumArray using " << TIMER_MSEC(NumArray) << " milliseconds";

    string input1 = R"([
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]])";
    string operations = "[sumRegion,sumRegion,sumRegion]";
    string args = "[[2, 1, 4, 3],[1, 1, 2, 2],[1, 2, 2, 4]]";
    string expectedOutputs = "[8,11,12]";
    util::Log(logESSENTIAL) << "Running NumMatrix tests:";
    TIMER_START(NumMatrix);
    NumMatrix_scaffold(input1, operations, args, expectedOutputs);
    TIMER_STOP(NumMatrix);
    util::Log(logESSENTIAL) << "NumMatrix using " << TIMER_MSEC(NumMatrix) << " milliseconds";

}
