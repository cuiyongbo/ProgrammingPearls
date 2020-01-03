#include "leetcode.h"

using namespace std;

class Solution {
public:
    int findCircleNum(vector<vector<int>>& M);
    int findCircleNum_DFS(vector<vector<int>>& M);
};

int Solution::findCircleNum(vector<vector<int>>& M)
{
    int size = M.size();
    DSU dsu(size);
    for(int row=0; row<size; ++row)
    {
        for(int col=0; col<size; ++col)
        {
            if(M[row][col])
                dsu.unionFunc(col, row);
        }
    }
    return dsu.groupCount();
}

int Solution::findCircleNum_DFS(vector<vector<int>>& M)
{
    int N = M.size();
    function<void(int)> dfs = [&](int row)
    {
        for (int i = 0; i < N; ++i)
        {
            if(!M[row][i]) continue;
            M[row][i] = M[i][row] = 0;
            dfs(i);
        }
    };

    int ans = 0;
    for (int row = 0; row < N; ++row)
    {
        if(!M[row][row]) continue;
        ++ans;
        dfs(row);
    }
    return ans;
}

int main()
{
    vector<vector<int>> M {
        {1,1,0},
        {1,1,0},
        {0,0,1}
    };

    Solution ss;
    //cout << ss.findCircleNum(M) << "\n";
    cout << ss.findCircleNum_DFS(M) << "\n";
}
