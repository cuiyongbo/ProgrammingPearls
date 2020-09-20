#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 112, 113, 437*/

class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum);
    vector<vector<int>> pathSum(TreeNode* root, int sum);
    int pathSum_437(TreeNode* root, int sum);
};

bool Solution::hasPathSum(TreeNode* root, int sum) {
/*
    Given a binary tree and a sum, determine if the tree has a root-to-leaf path 
    such that adding up all the values along the path equals the given sum.
    Note: A leaf is a node with no children.
*/

    if (root == nullptr) {
        return false;
    } else if (root->left == nullptr && root->right == nullptr) {
        return root->val == sum;
    } else {
        sum -= root->val;
        return hasPathSum(root->left, sum) || hasPathSum(root->right, sum);
    }
}

vector<vector<int>> Solution::pathSum(TreeNode* root, int sum) {
/*
    Given a binary tree and a sum, find all root-to-leaf paths where each 
    path's sum equals the given sum. Note: A leaf is a node with no children.
    Hint: dfs with backtrace
*/
    vector<int> path;
    vector<vector<int>> ans;
    function<void(TreeNode*, int)> dfs = [&] (TreeNode* node, int sum) {
        if(node == nullptr) {
            // do nothing
        } else if (node->left == nullptr && node->right == nullptr) {
            if(node->val == sum) {
                path.push_back(node->val);
                ans.push_back(path);
                path.pop_back();
            }
        } else {
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

int Solution::pathSum_437(TreeNode* root, int sum) {
/*
    You are given a binary tree in which each node contains an integer value.
    Find the number of paths that sum to a given value.
    The path does not need to start or end at the root or a leaf, 
    but it must go downwards (traveling only from parent nodes to child nodes).
    The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.
*/

    // number of path started from node whose path sum is equal to remaining
    function<int (TreeNode*, int)> dfs = [&] (TreeNode* node, int remaining) {
        if(node == nullptr) {
            return 0;
        } else {
            remaining -= node->val;
            return ((remaining == 0) ? 1 : 0) 
                    + dfs(node->left, remaining) 
                    + dfs(node->right, remaining);
        }
    };

    if (root == nullptr) {
        return 0;
    } else {
        return dfs(root, sum) 
                + pathSum_437(root->left, sum) 
                + pathSum_437(root->right, sum);
    }
}

void hasPathSum_scaffold(string input, int sum, bool expected) {
    TreeNode* t = stringToTreeNode(input);
    Solution ss;
    bool actual = ss.hasPathSum(t, sum);
    if (actual == expected) {
        util::Log(logINFO) << "Case (" << input << ", " << sum <<
                ", expectResult<" << expected << ">) passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << sum <<
                ", expectResult<" << expected << ">) failed";
    }
}

void pathSum_scaffold(string input, int sum, string expectedResult) {
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    vector<vector<int>> actual = ss.pathSum(t, sum);
    auto expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case (" << input << ", " << sum << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ") failed";
    }
}

void pathSum_437_scaffold(string input, int sum, int expectedResult) {
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    int actual = ss.pathSum_437(t, sum);
    if (actual == expectedResult) {
        util::Log(logESSENTIAL) << "Case (" << input << ", " << sum << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ", " << expectedResult << ") failed";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running hasPathSum tests:";
    TIMER_START(hasPathSum);
    hasPathSum_scaffold("[]", 0, false);
    hasPathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, true);
    TIMER_STOP(hasPathSum);
    util::Log(logESSENTIAL) << "Running hasPathSum tests uses" << TIMER_MSEC(hasPathSum) << "ms.";

    util::Log(logESSENTIAL) << "Running pathSum tests:";
    TIMER_START(pathSum);
    pathSum_scaffold("[1,1,1]", 2, "[[1,1],[1,1]]");
    pathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, "[[5,4,11,2]]");
    pathSum_scaffold("[]", 22, "[]");
    TIMER_STOP(pathSum);
    util::Log(logESSENTIAL) << "Running pathSum tests uses" << TIMER_MSEC(pathSum) << "ms.";

    util::Log(logESSENTIAL) << "Running pathSum_437 tests:";
    TIMER_START(pathSum_437);
    pathSum_437_scaffold("[]", 0,  0);
    pathSum_437_scaffold("[1,1,1]", 2, 2);
    pathSum_437_scaffold("[10,5,-3,3,2,null,11,3,-2,null,1]", 8, 3);
    pathSum_437_scaffold("[1,-2,-3,1,3,-2,null,-1]", -1, 4);
    TIMER_STOP(pathSum_437);
    util::Log(logESSENTIAL) << "Running pathSum_437 tests uses" << TIMER_MSEC(pathSum_437) << "ms.";    
}
