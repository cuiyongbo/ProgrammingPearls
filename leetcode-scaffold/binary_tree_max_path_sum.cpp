#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 124, 543, 687 */

class Solution
{
public:
    int maxPathSum(TreeNode* root);
    int diameterOfBinaryTree(TreeNode* root);
    int longestUnivaluePath(TreeNode* root);
};

int Solution::maxPathSum(TreeNode* root)
{
    /*
        Given a non-empty binary tree, find the maximum path sum.
        For this problem, a path is defined as any sequence of nodes from some
        starting node to any node in the tree along the parent-child connections.
        The path must contain at least one node and does not need to go through the root.

        helper(root) return max path sum
            1. starting from root and
            2. at most one child can be used (parent-child relation constraint).

    */

    int ans = INT_MIN;
    function<int(TreeNode*)> helper = [&](TreeNode* node)
    {
        if(node == NULL) return 0;

        int l = helper(node->left);
        int r = helper(node->right);
        int sum = node->val + std::max(0, l) + std::max(0, r);
        ans = std::max(ans, sum);
        return node->val + std::max(0, std::max(l, r));
    };

    helper(root);

    return ans;
}

int Solution::diameterOfBinaryTree(TreeNode* root)
{
    /*
        Given a binary tree, you need to compute the length of the diameter of the tree.
        The diameter of a binary tree is the length of the longest path between any two
        nodes in a tree. This path may or may not pass through the root.

        helper(root) return the node count of a longest path:
            path starting from root
            only one child can be used
    */

    int ans = 0;
    function<int(TreeNode*)> helper = [&](TreeNode* node)
    {
        if(node == NULL) return 0;

        int l = helper(node->left);
        int r = helper(node->right);
        int sum = l + r; // path length
        ans = std::max(ans, l+r);
        return 1 + std::max(l, r);
    };

    helper(root);

    return ans;
}

int Solution::longestUnivaluePath(TreeNode* root)
{
    /*
        Given a binary tree, find the length of the longest path
        where each node in the path has the same value.
        This path may or may not pass through the root.

        The length of a path between two nodes is represented
        by the number of edges between two nodes.

        helper(node) returns the lenght of longest univalue path:
            starting from node
            at most one child subtree can be used
    */

    int ans = 0;
    function<int(TreeNode*)> helper = [&](TreeNode* node)
    {
        if(node == NULL) return 0;

        int l = helper(node->left);
        int r = helper(node->right);

        int pl=0, pr=0;
        if(node->left != NULL && node->left->val == node->val)
            pl = l+1;
        if(node->right != NULL && node->right->val == node->val)
            pr = r+1;

        ans = std::max(ans, pl+pr);
        return std::max(pl, pr);
    };

    helper(root);

    return ans;
}

void maxPathSum_scaffold(string input, int expected)
{
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.maxPathSum(root);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expected << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed";
        util::Log(logERROR) << "expected: " << expected << ", actual: " << actual;
    }
}

void diameterOfBinaryTree_scaffold(string input, int expected)
{
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.diameterOfBinaryTree(root);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expected << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed";
        util::Log(logERROR) << "expected: " << expected << ", actual: " << actual;
    }
}

void longestUnivaluePath_scaffold(string input, int expected)
{
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.longestUnivaluePath(root);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expected << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed";
        util::Log(logERROR) << "expected: " << expected << ", actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running maxPathSum tests:";
    TIMER_START(maxPathSum);
    maxPathSum_scaffold("[-3]", -3);
    maxPathSum_scaffold("[1,2,3]", 6);
    maxPathSum_scaffold("[1,2,3,4]", 10);
    maxPathSum_scaffold("[-10,9,20,null,null,15,7]", 42);
    maxPathSum_scaffold("[-10,9,20,null,null,15,7,8]", 50);
    TIMER_STOP(maxPathSum);
    util::Log(logESSENTIAL) << "maxPathSum using " << TIMER_MSEC(maxPathSum) << " milliseconds";

    util::Log(logESSENTIAL) << "Running diameterOfBinaryTree tests:";
    TIMER_START(diameterOfBinaryTree);
    diameterOfBinaryTree_scaffold("[]", 0);
    diameterOfBinaryTree_scaffold("[1]", 0);
    diameterOfBinaryTree_scaffold("[1,2,3,4,5]", 3);
    diameterOfBinaryTree_scaffold("[-10,9,20,null,null,15,7]", 3);
    diameterOfBinaryTree_scaffold("[-10,9,20,null,null,15,7,8]", 4);
    TIMER_STOP(diameterOfBinaryTree);
    util::Log(logESSENTIAL) << "diameterOfBinaryTree using " << TIMER_MSEC(diameterOfBinaryTree) << " milliseconds";

    util::Log(logESSENTIAL) << "Running longestUnivaluePath tests:";
    TIMER_START(longestUnivaluePath);
    longestUnivaluePath_scaffold("[1,2,3,4,5]", 0);
    longestUnivaluePath_scaffold("[5,5,5,5]", 3);
    TIMER_STOP(longestUnivaluePath);
    util::Log(logESSENTIAL) << "longestUnivaluePath using " << TIMER_MSEC(longestUnivaluePath) << " milliseconds";
}
