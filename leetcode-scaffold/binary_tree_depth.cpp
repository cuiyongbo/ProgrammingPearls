#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 104, 110, 111 */
class Solution {
public:
    int maxDepth(TreeNode* root);
    int minDepth(TreeNode* root);
    int maxDepth_recursive(TreeNode* root);
    int maxDepth_iterative(TreeNode* root);
    bool isBalanced(TreeNode* root);
};

bool Solution::isBalanced(TreeNode* root) {
/*
Given a binary tree, determine if it is height-balanced.
A height-balanced binary tree is defined as: a binary tree in which the left and 
right subtrees of every node differ in height by no more than 1.
*/

    if (root == nullptr) {
        return true;
    } else {
        int l = maxDepth(root->left);
        int r = maxDepth(root->right);
        if (std::abs(l-r) > 1) {
            return false;
        } else {
            return isBalanced(root->left) && isBalanced(root->right);
        }
    }
}

int Solution::maxDepth(TreeNode* root) {
/*
    Given a binary tree, find its maximum depth.
    The maximum depth is the number of nodes along 
    the longest path from the root node down to the farthest leaf node.
    Note: A leaf is a node with no children.
*/

    return maxDepth_iterative(root);
}

int Solution::maxDepth_recursive(TreeNode* root) {
    if (root == nullptr) {
        return 0;
    }
    return std::max(maxDepth_recursive(root->left),
                    maxDepth_recursive(root->right)) + 1;
}

int Solution::maxDepth_iterative(TreeNode* root) {
    if (root == nullptr) {
        return 0;
    }
    int depth = 0;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right) {
                q.push(t->right);
            }
        }
        depth++;
    }
    return depth;
}

int Solution::minDepth(TreeNode* root)
{
    /*
        The minimum depth is the number of nodes along the shortest path 
        from the root node down to the nearest leaf node.
        Hint: find the earliest leaf node.
    */
    if (root == nullptr) {
        return 0;
    }

    int depth = 0;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        depth++;
        int size = q.size();
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            if(t->left == nullptr && t->right == nullptr) {
                return depth;
            }
            if(t->left != nullptr) {
                q.push(t->left);
            }
            if(t->right != nullptr) {
                q.push(t->right);
            }
        }
    }
    return depth;
}

void maxDepth_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    auto expected = std::stoi(input2);
    int ans = 0;
    Solution ss;
    ans = ss.maxDepth_iterative(root);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Iterative Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Iterative Case(" << input1 << ", " << input2 << ") failed, "
                            << "actual: " << ans << ", expected: " << input2;
    }

    ans = ss.maxDepth_recursive(root);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Recursive Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Recursive Case(" << input1 << ", " << input2 << ") failed, "
                            << "actual: " << ans << ", expected: " << input2;
    }
}

void minDepth_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    auto expected = std::stoi(input2);
    int ans = 0;
    Solution ss;
    ans = ss.minDepth(root);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed, "
                            << "actual: " << ans << ", expected: " << input2;
    }
}

void isBalanced_scaffold(string input1, bool expected) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    bool ans = ss.isBalanced(root);
    if (ans == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expected << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expected << ") failed.";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running maxDepth tests:";
    TIMER_START(maxDepth);
    maxDepth_scaffold("[]", "0");
    maxDepth_scaffold("[1]", "1");
    maxDepth_scaffold("[1,2,3,4,5]", "3");
    maxDepth_scaffold("[3,9,20,null,null,15,7]", "3");
    TIMER_STOP(maxDepth);
    util::Log(logESSENTIAL) << "maxDepth tests using " << TIMER_MSEC(maxDepth) <<"ms";

    util::Log(logESSENTIAL) << "Running minDepth tests:";
    TIMER_START(minDepth);
    minDepth_scaffold("[]", "0");
    minDepth_scaffold("[1]", "1");
    minDepth_scaffold("[1,2,3,4,5]", "2");
    minDepth_scaffold("[3,9,20,null,null,15,7]", "2");
    TIMER_STOP(minDepth);
    util::Log(logESSENTIAL) << "minDepth tests using " << TIMER_MSEC(minDepth) <<"ms";    

    util::Log(logESSENTIAL) << "Running isBalanced tests:";
    TIMER_START(isBalanced);
    isBalanced_scaffold("[]", true);
    isBalanced_scaffold("[1]", true);
    isBalanced_scaffold("[1,2,3,4,5]", true);
    isBalanced_scaffold("[3,9,20,null,null,15,7]", true);
    isBalanced_scaffold("[1,2,2,3,3,null,null,4,4]", false);
    TIMER_STOP(isBalanced);
    util::Log(logESSENTIAL) << "isBalanced tests using " << TIMER_MSEC(isBalanced) <<"ms"; 

    return 0;
}
