#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 814, 669, 1325*/

class Solution {
public:
    TreeNode* pruneTree(TreeNode* root);
    TreeNode* trimBST(TreeNode* root, int L, int R);
    TreeNode* removeLeafNodes(TreeNode* root, int target);
};

TreeNode* Solution::pruneTree(TreeNode* root) {
/*
    We are given the head node root of a binary tree, where additionally every node's value is either a 0 or a 1. Return the same tree where every subtree (of the given tree) not containing a 1 has been removed. (Recall that the subtree of a node X is X, plus every node that is a descendant of X.)
    Hint: prune all subtrees which containes only 0
    Example:
        Input: [1,null,0,0,1]
        Output: [1,null,0,null,1]
*/
    // return the tree root at node with 0 nodes removed
    // traverse the tree in post-order
{    
    std::function<TreeNode*(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return node;
        }
        node->left = dfs(node->left);
        node->right = dfs(node->right);
        if (node->is_leaf() && node->val == 0) {
            node = nullptr;
        }
        return node;
    };
    return dfs(root);
}

if (0) { // iterative solution
    function<bool(TreeNode*)> dfs = [&] (TreeNode* node) { // preorder traversal
        if (node == nullptr) {
            return true;
        }
        return node->val == 0 && 
                dfs(node->left) && 
                dfs(node->right);
    };
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        for (int i=q.size(); i!=0; --i) {
            auto t = q.front(); q.pop();
            if (t == nullptr) {
                continue;
            }
            if (dfs(t->left)) {
                t->left = nullptr;
            } else {
                q.push(t->left);
            }
            if (dfs(t->right)) {
                t->right = nullptr;
            } else {
                q.push(t->right);
            }
        }
    }
    return root;
}

}


TreeNode* Solution::trimBST(TreeNode* root, int L, int R) {
/*
    Given the root of a binary search tree and the lowest and highest boundaries as low and high, 
    trim the tree so that all its elements lies in [low, high].
    You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.
*/
    if (root == nullptr) { // trivial case
        return nullptr;
    } else if (root->val > R) {
        return trimBST(root->left, L, R);
    } else if (root->val < L) {
        return trimBST(root->right, L, R);
    } else { // postorder traversal
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
    if (root == nullptr) { // trivial case
        return root;
    } else { // post-order traversal
        root->left = removeLeafNodes(root->left, target);
        root->right = removeLeafNodes(root->right, target);
        if (root->is_leaf() && root->val == target) {
            root = nullptr;
        }
        return root;
    }
}


void prune_tree_scaffold(std::string input1, std::string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    Solution ss;
    TreeNode* ans = ss.pruneTree(t1);
    if (binaryTree_equal(ans, t2)) {
        SPDLOG_INFO("Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed. actual: ", input1, input2);
        printBinaryTree(ans);
    }
}


void trim_bst_scaffold(std::string input1, int L, int R, std::string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    Solution ss;
    TreeNode* ans = ss.trimBST(t1, L, R);
    if (binaryTree_equal(ans, t2)) {
        SPDLOG_INFO("Case({}, {}, {}, {}) passed", input1, L, R, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}, {}) failed. actual:", input1, L, R, input2);
        printBinaryTree(ans);
    }
}


void remove_leaf_scaffold(std::string input1, int target, std::string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    Solution ss;
    TreeNode* ans = ss.removeLeafNodes(t1, target);
    if (binaryTree_equal(ans, t2)) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, target, input2);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed. actual: ", input1, target, input2);
        printBinaryTree(ans);
    }
}


int main() {
    SPDLOG_WARN("Running prune_tree tests:");
    TIMER_START(prune_tree);
    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,null,0,0,1]", "[1,null,0,null,1]");
    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,1,0,1,1,0,1,0]", "[1,1,0,1,1,null,1]");
    prune_tree_scaffold("[0]", "[]");
    prune_tree_scaffold("[0,0]", "[]");
    prune_tree_scaffold("[0,0,0]", "[]");
    TIMER_STOP(prune_tree);
    SPDLOG_WARN("prune_tree tests use {} ms", TIMER_MSEC(prune_tree));

    SPDLOG_WARN("Running trim_bst tests:");
    TIMER_START(trim_bst);
    trim_bst_scaffold("[1,0,2]", 0, 2, "[1,0,2]");
    trim_bst_scaffold("[1,0,2]", 1, 2, "[1,null,2]");
    trim_bst_scaffold("[3,0,4,null,2,null,null,1]", 1, 3, "[3,2,null,1]");
    trim_bst_scaffold("[1]", 1, 3, "[1]");
    trim_bst_scaffold("[1,null,2]", 1, 3, "[1,null,2]");
    trim_bst_scaffold("[1,null,2]", 2, 4, "[2]");
    TIMER_STOP(trim_bst);
    SPDLOG_WARN("trim_bst tests use {} ms", TIMER_MSEC(trim_bst));

    SPDLOG_WARN("Running remove_leaf tests:");
    TIMER_START(remove_leaf);
    remove_leaf_scaffold("[1,3,3,3,2]", 3, "[1,3,null,null,2]");
    remove_leaf_scaffold("[1,2,3,2,null,2,4]", 2, "[1,null,3,null,4]");
    remove_leaf_scaffold("[1,2,null,2,null,2]", 2, "[1]");
    remove_leaf_scaffold("[1,1,1]", 1, "[]");
    TIMER_STOP(remove_leaf);
    SPDLOG_WARN("remove_leaf tests use {} ms", TIMER_MSEC(remove_leaf));
}
