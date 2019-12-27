#include "leetcode.h"

using namespace std;

class Solution {
public:
    int longestCommonSubsequence(string x, string y)
    {
        int m = x.size();
        int n = y.size();

        // dp[i][j] means the longest common subsequence of x[0, i) and y[0, j)
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

        for(int i=0; i<m; ++i)
        {
            for (int j=0; j<n; ++j)
            {
                if(x[i] == y[j])
                    dp[i+1][j+1] = dp[i][j] + 1;
                else
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]);
            }
        }
        return dp[m][n];
    }
};

int main()
{
    string x = "xxxyyy";
    string y = "xxxyyy";
    Solution ss;
    cout << ss.longestCommonSubsequence(x, y) << "\n";
}
