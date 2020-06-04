#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 129, 257*/

class Solution
{
public:
    int sumNumbers(TreeNode* root);
    vector<string> binaryTreePaths(TreeNode* root);
};

int Solution::sumNumbers(TreeNode* root)
{
    function<int(TreeNode*, int)> dfs = [&](TreeNode* node, int curSum)
    {
        if(node == NULL) return 0;

        curSum = curSum*10 + node->val;
        if (node->left == NULL && node->right == NULL)
        {
            return curSum;
        }
        else
        {
            return dfs(node->left, curSum) + dfs(node->right, curSum);
        }
    };

    return dfs(root, 0);
}

vector<string> Solution::binaryTreePaths(TreeNode* root)
{
    vector<string> ans;
    function<void(TreeNode*,string)> dfs = [&](TreeNode* node, string path)
    {
        if(node == NULL)
        {
            return;
        }

        if(!path.empty())
        {
            path += "->";
        }

        path += to_string(node->val);
        if(node->left == NULL && node->right == NULL)
        {
            ans.emplace_back(path);
        }
        else
        {
            dfs(node->left, path);
            dfs(node->right, path);
        }
    };

    string path;
    dfs(root, path);
    return ans;
}

void sumNumbers_scaffold(string input, int expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.sumNumbers(root);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed";
        util::Log(logERROR) << "expected: " << expectedResult << ", actual: " << actual;
    }

    destroyBinaryTree(root);
}


void binaryTreePaths_scaffold(string input, vector<string>& expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    vector<string> actual = ss.binaryTreePaths(root);
    if(actual.size() == expectedResult.size() && equal(actual.begin(), actual.end(), expectedResult.begin()))
    {
        util::Log(logESSENTIAL) << "Case(" << input << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ") failed";

        util::Log(logERROR) << "expected:";
        for(auto& s: expectedResult)
            util::Log(logERROR) << s;

        util::Log(logERROR) << "acutal:";
        for(auto& s: actual)
            util::Log(logERROR) << s;
    }

    destroyBinaryTree(root);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    sumNumbers_scaffold("[1,2,3]", 25);
    sumNumbers_scaffold("[1,2,3,4,6,null,8]", 388);

    vector<string> expected;
    expected.emplace_back("1->2");
    expected.emplace_back("1->3");
    binaryTreePaths_scaffold("[1,2,3]", expected);

    expected.clear();
    expected.emplace_back("1->2->4");
    expected.emplace_back("1->2->6");
    expected.emplace_back("1->3->8");
    binaryTreePaths_scaffold("[1,2,3,4,6,null,8]", expected);
}
