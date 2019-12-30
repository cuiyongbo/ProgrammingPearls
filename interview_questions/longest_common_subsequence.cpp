#include "leetcode.h"

using namespace std;

class Solution {
public:
    int longestCommonSubsequence_length(string x, string y);
    int longestCommonSubsequence_length_space_opt_1(string x, string y);
    int longestCommonSubsequence_length_space_opt_2(string x, string y);
    string longestCommonSubsequence(string x, string y);
};

int Solution::longestCommonSubsequence_length(string x, string y)
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

int Solution::longestCommonSubsequence_length_space_opt_1(string x, string y)
{
    int m = x.size();
    int n = y.size();

    if(m < n)
    {
        swap(x, y);
        swap(m, n);
    }

    vector<vector<int>> dp(2, vector<int>(n+1, 0));
    for(int i=0; i<m; ++i)
    {
        for (int j=0; j<n; ++j)
        {
            if(x[i] == y[j])
                dp[1][j+1] = dp[0][j] + 1;
            else
                dp[1][j+1] = max(dp[1][j], dp[0][j+1]);
        }
        dp[0].assign(dp[1].begin(), dp[1].end());
    }
    return dp[1][n];
}

int Solution::longestCommonSubsequence_length_space_opt_2(string x, string y)
{
    int m = x.size();
    int n = y.size();

    if(m < n)
    {
        swap(x, y);
        swap(m, n);
    }

    vector<int> dp(n+1, 0);
    for(int i=1; i<=m; ++i)
    {
        int prev = 0;  // dp[i-1][j-1]
        for (int j=1; j<=n; ++j)
        {
            int cur = dp[j]; // dp[i-1][j]
            if(x[i-1] == y[j-1])
            {
                dp[j] = prev + 1;
            }
            else
            {
                dp[j] = max(cur, dp[j-1]);
            }
            prev = cur;
        }
    }
    return dp[n];
}

string Solution::longestCommonSubsequence(string x, string y)
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

    string ans;
    int row = m, col = n;
    while(row > 0 && col > 0)
    {
        if(x[row-1] == y[col-1])
        {
            ans.push_back(x[row-1]);
            row--; col--;
        }
        else if(dp[row-1][col] >= dp[row][col-1])
        {
            row--;
        }
        else
        {
            col--;
        }
    }
    reverse(ans.begin(), ans.end());
    return ans;
}

int main()
{
    string x = "xaxyyy";
    string y = "xxaxbyy";
    Solution ss;
    //cout << ss.longestCommonSubsequence_length(x, y) << "\n";
    //cout << ss.longestCommonSubsequence_length_space_opt_1(x, y) << "\n";
    cout << ss.longestCommonSubsequence_length_space_opt_2(x, y) << "\n";
    cout << ss.longestCommonSubsequence(x, y) << "\n";
}
