#include "leetcode.h"

using namespace std;

class Solution {
public:
    int minDistance(string word1, string word2);
};

int Solution::minDistance(string word1, string word2)
{
    int n1 = word1.size();
    int n2 = word2.size();

    // dp[i][j] means minDistance(word1[0:i], word2[0:j])
    vector<vector<int>> dp(n1+1, vector<int>(n2+1));

    for(int i=0; i<=n1; i++) dp[i][0] = i;
    for(int i=0; i<=n2; i++) dp[0][i] = i;
    for(int i=1; i<=n1; i++)
    {
        for (int j=0; j<=n2; j++)
        {
            // penalty == 0 means no operation
            // penalty == 1 means taking an replace operation
            // dp[i-1][j] means taking an delete operation
            // dp[i][j-1] means taking an insert operation
            int penalty = (word1[i-1] == word2[j-1]) ? 0 : 1;
            dp[i][j] = min(dp[i-1][j-1] + penalty, min(dp[i-1][j], dp[i][j-1]) + 1);
        }
    }

    return dp[n1][n2];
}

int main()
{
    string w1 = "horse";
    string w2 = "hope";

    Solution ss;
    int n = ss.minDistance(w1, w2);
    printf("minDistance(%s, %s): %d\n", w1.c_str(), w2.c_str(), n);
}
