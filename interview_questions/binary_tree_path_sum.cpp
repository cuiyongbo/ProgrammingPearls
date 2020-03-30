#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 112, 113, 437*/

class Solution
{
public:
    bool hasPathSum(TreeNode* root, int sum);
    vector<vector<int>> pathSum(TreeNode* root, int sum);
    int pathSum_437(TreeNode* root, int sum);
};

bool Solution::hasPathSum(TreeNode* root, int sum)
{
    if(root == NULL)
    {
        return false;
    }
    else if(root->left == NULL && root->right == NULL)
    {
        return root->val == sum;
    }
    else
    {
        sum -= root->val;
        return hasPathSum(root->left, sum) || hasPathSum(root->right, sum);
    }
}

vector<vector<int>> Solution::pathSum(TreeNode* root, int sum)
{
    vector<int> path;
    vector<vector<int>> ans;
    function<void(TreeNode*, int)> dfs = [&] (TreeNode* node, int sum)
    {
        if(node == NULL)
        {
            // do nothing
        }
        else if(node->left == NULL && node->right == NULL)
        {
            if(node->val == sum)
            {
                path.push_back(node->val);
                ans.push_back(path);
                path.pop_back();
            }
        }
        else
        {
            sum -= node->val;
            path.push_back(node->val);

            dfs(node->left, sum);
            dfs(node->right, sum);

            path.pop_back();
        }
    };

    dfs(root, sum);

    return ans;
}

int Solution::pathSum_437(TreeNode* root, int sum)
{
    function<int (TreeNode*, int)> dfs = [&] (TreeNode* node, int remaining)
    {
        if(node == NULL)
        {
            return 0;
        }
        else
        {
            remaining -= node->val;
            return ((remaining == 0) ? 1 : 0) + dfs(node->left, remaining) + dfs(node->right, remaining);
        }
    };

    if (root == NULL) return 0;
    return dfs(root, sum) + pathSum_437(root->left, sum) + pathSum_437(root->right, sum);
}

void hasPathSum_scaffold(string input, int sum, bool expected)
{
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    bool actual = ss.hasPathSum(t, sum);
    if (actual == expected)
    {
        util::Log(logESSENTIAL) << "Case (" << input << ", " << sum <<
                ", expectResult<" << expected << ">) passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ", " << sum <<
                ", expectResult<" << expected << ">) failed";
    }

    destroyBinaryTree(t);
}

void pathSum_scaffold(string input, int sum, vector<vector<int>>& expectedResult)
{
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    vector<vector<int>> actual = ss.pathSum(t, sum);

    if (equal(actual.begin(), actual.end(), expectedResult.begin()))
    {
        util::Log(logESSENTIAL) << "Case (" << input << ", " << sum << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ") failed";
    }

    destroyBinaryTree(t);
}

void pathSum_437_scaffold(string input, int sum, int expectedResult)
{
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    int actual = ss.pathSum_437(t, sum);
    if (actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input << ", " << sum << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ", " << expectedResult << ") failed";
    }

    destroyBinaryTree(t);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    hasPathSum_scaffold("[]", 0, false);
    hasPathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, true);

    vector<vector<int>> expected {{1,1}, {1,1}};
    pathSum_scaffold("[1,1,1]", 2,  expected);

    expected.clear();
    expected.push_back({5,4,11,2});
    expected.push_back({5,8,4,5});
    pathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,5,1]", 22,  expected);

    expected.clear();
    pathSum_scaffold("[]", 22,  expected);

    pathSum_437_scaffold("[]", 0,  0);
    pathSum_437_scaffold("[1,1,1]", 2, 2);
    pathSum_437_scaffold("[10,5,-3,3,2,null,11,3,-2,null,1]", 8, 3);
    pathSum_437_scaffold("[1,-2,-3,1,3,-2,null,-1]", -1, 4);
}
