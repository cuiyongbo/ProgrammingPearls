#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 98, 530 */

class Solution
{
public:
    bool isValidBST(TreeNode* root);
    int getMinimumDifference(TreeNode* root);
};

bool Solution::isValidBST(TreeNode* root)
{
    function<bool(TreeNode*, TreeNode*, TreeNode*)>dfs = [&](TreeNode* node, TreeNode* maxNode, TreeNode* minNode)
    {
        if(node == NULL)
        {
            return true;
        }
        else if(maxNode != NULL && maxNode->val <= node->val)
        {
            return false;
        }
        else if(minNode != NULL && minNode->val >= node->val)
        {
            return false;
        }
        else
        {
            return dfs(node->left, node, minNode) && dfs(node->right, maxNode, node);
        }
    };

    return dfs(root, NULL, NULL);
}

int Solution::getMinimumDifference(TreeNode* root)
{
    /*
        Given a binary search tree with non-negative values,
        find the minimum absolute difference between values of any two nodes.
    */

    vector<int> inorderSeq;
    function<void(TreeNode*)> dfs = [&](TreeNode* node)
    {
        if(node == NULL) return;
        dfs(node->left);
        inorderSeq.push_back(node->val);
        dfs(node->right);
    };

    dfs(root);

    int minDiff = INT_MAX;
    for(int i=1; i<inorderSeq.size(); ++i)
    {
        minDiff = min(minDiff, abs(inorderSeq[i] - inorderSeq[i-1]));
    }
    return minDiff;
}

void isValidBST_scaffold(string input, bool expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    bool actual = ss.isValidBST(root);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input << ", expected <" << expectedResult << ">) passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ", expected<" << expectedResult << ">) failed";
    }

    destroyBinaryTree(root);
}

void getMinimumDifference_scaffold(string input, int expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.getMinimumDifference(root);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ", expected: " << expectedResult << ") failed";
    }
    destroyBinaryTree(root);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isValidBST tests:";
    TIMER_START(isValidBST);
    isValidBST_scaffold("[1,2,3]", false);
    isValidBST_scaffold("[2,1,3]", true);
    isValidBST_scaffold("[4,3,5,null,2]", false);
    TIMER_STOP(isValidBST);
    util::Log(logESSENTIAL) << "isValidBST using " << TIMER_MSEC(isValidBST) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running getMinimumDifference tests:";
    TIMER_START(getMinimumDifference);
    getMinimumDifference_scaffold("[1,2,3]", 1);
    getMinimumDifference_scaffold("[5,2,6,0,4,null,8]", 1);
    TIMER_STOP(getMinimumDifference);
    util::Log(logESSENTIAL) << "getMinimumDifference using " << TIMER_MSEC(getMinimumDifference) << " milliseconds.";
}
