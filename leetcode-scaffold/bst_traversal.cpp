#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 98, 530 */

class Solution {
public:
    bool isValidBST(TreeNode* root);
    int getMinimumDifference(TreeNode* root);
};


/*
    Given a binary tree, determine if it is a valid binary search tree (BST).
    Assume a BST is defined as follows:
        The left subtree of a node contains only nodes with keys less than the node’s key.
        The right subtree of a node contains only nodes with keys greater than the node’s key.
        Both the left and right subtrees must also be binary search trees.
    Hint: if the bst is valid, then the inorder traversal sequence would be sorted in ascending order
*/
bool Solution::isValidBST(TreeNode* root) {
    TreeNode* predecessor = nullptr;
    function<bool(TreeNode*)> dfs = [&] (TreeNode* node) { // inorder traversal
        if (node == nullptr) { // trivial case
            return true;
        }
        if (!dfs(node->left)) {
            return false;
        }
        if (predecessor != nullptr && predecessor->val >= node->val) {
            return false;
        }
        predecessor = node;
        return dfs(node->right);
    };
    return dfs(root);
}


/*
    Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.
    Hint: perform an inorder traversal to find the difference between current node and its predecessor.
*/
int Solution::getMinimumDifference(TreeNode* root) {
    int ans = INT32_MAX;
    TreeNode* predecessor = nullptr;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        dfs(node->left);
        if (predecessor != nullptr) {
            ans = std::min(ans, node->val-predecessor->val);
        }
        predecessor = node;
        dfs(node->right);
    };
    dfs(root);
    return ans;
}


void isValidBST_scaffold(std::string input, bool expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    bool actual = ss.isValidBST(root);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case (" << input << ", expected <" << expectedResult << ">) passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", expected<" << expectedResult << ">) failed, actual: " << actual;
    }
}


void getMinimumDifference_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.getMinimumDifference(root);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case (" << input << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", expected: " << expectedResult << ") failed, actual: " << actual;
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running isValidBST tests:";
    TIMER_START(isValidBST);
    isValidBST_scaffold("[1,2,3]", false);
    isValidBST_scaffold("[2,1,3]", true);
    isValidBST_scaffold("[5,1,4,null,null,3,6]", false);
    isValidBST_scaffold("[4,3,5,null,2]", false);
    TIMER_STOP(isValidBST);
    util::Log(logESSENTIAL) << "isValidBST using " << TIMER_MSEC(isValidBST) << "ms.";

    util::Log(logESSENTIAL) << "Running getMinimumDifference tests:";
    TIMER_START(getMinimumDifference);
    getMinimumDifference_scaffold("[2,1,3]", 1);
    getMinimumDifference_scaffold("[5,2,6,0,4,null,8]", 1);
    getMinimumDifference_scaffold("[4,2,6,1,3]", 1);
    getMinimumDifference_scaffold("[1,0,48,null,null,12,49]", 1);
    TIMER_STOP(getMinimumDifference);
    util::Log(logESSENTIAL) << "getMinimumDifference using " << TIMER_MSEC(getMinimumDifference) << "ms.";
}
