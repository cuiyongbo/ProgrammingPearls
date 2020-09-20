#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 814, 669, 1325*/

class Solution {
public:
    TreeNode* pruneTree(TreeNode* root);
    bool isSameTree(TreeNode* p, TreeNode* q);
    TreeNode* trimBST(TreeNode* root, int L, int R);
    TreeNode* removeLeafNodes(TreeNode* root, int target);
};

TreeNode* Solution::pruneTree(TreeNode* root) {
/*
We are given the head node root of a binary tree, where additionally every node's value is either a 0 or a 1.
Return the same tree where every subtree (of the given tree) not containing a 1 has been removed.
(Recall that the subtree of a node X is X, plus every node that is a descendant of X.)
Example:
    Input: [1,null,0,0,1]
    Output: [1,null,0,null,1]
*/

    function<bool(TreeNode*)> allNodeZeros = [&] (TreeNode* root) {
        if (root == nullptr) {
            return true;
        } else if (root->val != 0) {
            return false;
        } else {
            return allNodeZeros(root->left) && allNodeZeros(root->right);
        }
    };

    if (root == nullptr) {
        return nullptr;
    } else if (root->val == 1) {
        root->left = pruneTree(root->left);
        root->right = pruneTree(root->right);
        return root;
    } else {
        root->left = allNodeZeros(root->left) ? nullptr : root->left;
        root->right = allNodeZeros(root->right) ? nullptr : root->right;
        if (root->left == nullptr && root->right == nullptr) {
            return nullptr;
        } else {
            return root;
        }
    }
}

bool Solution::isSameTree(TreeNode* p, TreeNode* q) {
    if (p == nullptr && q == nullptr) {
        return true;
    } else if (p == nullptr || q == nullptr) {
        return false;
    } else if (p->val != q->val) {
        return false;
    } else {
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
}

TreeNode* Solution::trimBST(TreeNode* root, int L, int R) {
/*
    Given the root of a binary search tree and the lowest and highest boundaries as low and high, 
    trim the tree so that all its elements lies in [low, high]. You might need to change the root of 
    the tree, so the result should return the new root of the trimmed binary search tree.
*/
    if (root == nullptr) {
        return root;
    } else if (root->val < L) {
        return trimBST(root->right, L, R);
    } else if (root->val > R) {
        return trimBST(root->left, L, R);
    } else {
        root->left = trimBST(root->left, L, R);
        root->right = trimBST(root->right, L, R);
        return root;
    }
}

TreeNode* Solution::removeLeafNodes(TreeNode* root, int target) {
/*
    Given a binary tree root and an integer target, delete all the *leaf nodes* with value target.
    Note that once you delete a leaf node with value target, if it's parent node becomes a leaf 
    node and has the value target, it should also be deleted (you need to continue doing that until you can't).
    Example:
        Input: root = [1,2,3,2,null,2,4], target = 2
        Output: [1,null,3,null,4]
*/

    if (root == nullptr) {
        return nullptr;
    } else {
        root->left = removeLeafNodes(root->left, target);
        root->right = removeLeafNodes(root->right, target);
        if (root->val == target && root->left == nullptr && root->right == nullptr) {
            return nullptr;
        } else {
            return root;
        }
    }
}

void prune_tree_scaffold(string input1, string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.pruneTree(t1);
    if (ss.isSameTree(ans, t2)) {
        util::UnbufferedLog(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
    }
}

void trim_bst_scaffold(string input1, int L, int R, string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.trimBST(t1, L, R);
    if (ss.isSameTree(ans, t2)) {
        util::UnbufferedLog(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
    }
}

void remove_leaf_scaffold(string input1, int target, string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.removeLeafNodes(t1, target);
    if (ss.isSameTree(ans, t2)) {
        util::UnbufferedLog(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running prune_tree tests:";
    TIMER_START(prune_tree);
    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,null,0,0,1]", "[1,null,0,null,1]");
    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,1,0,1,1,0,1,0]", "[1,1,0,1,1,null,1]");
    TIMER_STOP(prune_tree);
    util::Log(logESSENTIAL) << "Running prune_tree tests uses " << TIMER_MSEC(prune_tree) << "ms.";

    util::Log(logESSENTIAL) << "Running trim_bst tests:";
    TIMER_START(trim_bst);
    trim_bst_scaffold("[1,0,2]", 0, 2, "[1,0,2]");
    trim_bst_scaffold("[1,0,2]", 1, 2, "[1,null,2]");
    trim_bst_scaffold("[3,0,4,null,2,null,null,1]", 1, 3, "[3,2,null,1]");
    trim_bst_scaffold("[1]", 1, 3, "[1]");
    trim_bst_scaffold("[1,null,2]", 1, 3, "[1,null,2]");
    trim_bst_scaffold("[1,null,2]", 2, 4, "[2]");
    TIMER_STOP(trim_bst);
    util::Log(logESSENTIAL) << "Running trim_bst tests uses " << TIMER_MSEC(trim_bst) << "ms.";

    util::Log(logESSENTIAL) << "Running remove_leaf tests:";
    TIMER_START(remove_leaf);
    remove_leaf_scaffold("[1,3,3,3,2]", 3, "[1,3,null,null,2]");
    remove_leaf_scaffold("[1,2,3,2,null,2,4]", 2, "[1,null,3,null,4]");
    remove_leaf_scaffold("[1,2,null,2,null,2]", 2, "[1]");
    remove_leaf_scaffold("[1,1,1]", 1, "[]");
    TIMER_STOP(remove_leaf);
    util::Log(logESSENTIAL) << "Running remove_leaf tests uses " << TIMER_MSEC(remove_leaf) << "ms.";
}
