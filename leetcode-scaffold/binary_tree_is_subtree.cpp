#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 100, 101, 572 */
class Solution {
public:
    bool isSameTree(TreeNode* node, TreeNode* t);
    bool isSubtree(TreeNode* s, TreeNode* t); 
    bool isSymmetric(TreeNode* root);
};

bool Solution::isSameTree(TreeNode* l, TreeNode* r) {
/*
    Given two binary trees, write a function to check if they are the same or not.
    Two binary trees are considered the same if they are structurally identical and the nodes have the same value.
*/

{ // recursive solution
    if (l==nullptr && r==nullptr) { // trivial case
        return true;
    } else if (l==nullptr || r==nullptr) { // trivial case
        return false;
    } else if (l->val != r->val) { // trivial case
        return false;
    } else {
        return isSameTree(l->left, r->left) && isSameTree(l->right, r->right);
    }
}


{ // iterative solution
    std::queue<TreeNode*> q;
    q.push(l); q.push(r);
    while (!q.empty()) {
        auto t1 = q.front(); q.pop();
        auto t2 = q.front(); q.pop();
        if (t1 == nullptr && t2 == nullptr) {
            continue;
        } else if (t1 == nullptr || t2 == nullptr) {
            return false;
        } else if (t1->val != t2->val) {
            return false;
        } else {
            q.push(t1->left); q.push(t2->left);
            q.push(t1->right); q.push(t2->right);
        }
    }
    return true;
}

}

bool Solution::isSubtree(TreeNode* s, TreeNode* t)  {
/*
    Given two non-empty binary trees s and t, check whether tree t 
    has exactly the same structure and node values with a subtree of s. 
    A subtree of s is a tree consists of a node in s and all of this node's descendants. 
    The tree s could also be considered as a subtree of itself.
*/

if (0) { // recursive solution
    return isSameTree(s, t) ||
            (s!=nullptr && isSameTree(s->left, t)) ||
            (s!=nullptr && isSameTree(s->right, t));
}

{ // iterative solution
    std::queue<TreeNode*> q; q.push(s);
    while (!q.empty()) {
        auto node = q.front(); q.pop();
        if (isSameTree(node, t)) {
            return true;
        }
        if (node != nullptr) {
            q.push(node->left);
            q.push(node->right);
        }
    }
    return false;
}

}


bool Solution::isSymmetric(TreeNode* root) {
/*
    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
    For example, this binary tree [1,2,2,3,4,4,3] is symmetric.
*/

if (1) { // recursive solution
    function<bool(TreeNode*, TreeNode*)> dfs = [&] (TreeNode* l, TreeNode* r) {
        if (l==nullptr && r==nullptr) {
            return true;
        } else if (l==nullptr || r==nullptr) {
            return false;
        } else if (l->val != r->val) {
            return false;
        } else {
            return dfs(l->left, r->right) && dfs(r->left, l->right);
        }
    };

    // spare one level function call
    if (root == nullptr) {
        return true;
    } else {
        return dfs(root->left, root->right);
    }
    // return dfs(root, root);
}

{ // iterative solution 
    if (root == nullptr) {
        return true;
    }
    std::queue<TreeNode*> q;
    q.push(root->left); q.push(root->right);
    while (!q.empty()) {
        auto t1 = q.front(); q.pop();
        auto t2 = q.front(); q.pop();
        if (t1 == nullptr && t2 == nullptr) {
            continue;
        } else if (t1==nullptr || t2==nullptr) {
            return false;
        } else if (t1->val != t2->val) {
            return false;
        } else {
            // push node according to symmetry
            q.push(t1->left); q.push(t2->right);
            q.push(t1->right); q.push(t2->left);
        }
    }
    return true;
}

}


void isSymmetric_scaffold(std::string input1, bool expected) {
    TreeNode* root = stringToTreeNode(input1);
    std::unique_ptr<TreeNode> guard(root);
    Solution ss;
    bool actual = ss.isSymmetric(root);
    if (actual == expected) {
        SPDLOG_INFO( "Case({}, expected={}) passed", input1, expected);
    } else {
        SPDLOG_ERROR( "Case({}, expected={}) failed. actual={}", input1, expected, actual);
    }
}


void isSameTree_scaffold(std::string input1, std::string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);
    Solution ss;
    bool actual = ss.isSameTree(t1, t2);
    bool expected = input1 == input2;
    if (expected == actual) {
        SPDLOG_INFO( "Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR( "Case({}, {}) failed. expected={}, actual={}", input1, input2, expected, actual);
    }
}


void isSubtree_scaffold(std::string input1, std::string input2, bool expected) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);
    Solution ss;
    bool actual = ss.isSubtree(t1, t2);
    if (expected == actual) {
        SPDLOG_INFO( "Case({}, {}) passed", input1, input2);
    } else {
        SPDLOG_ERROR( "Case({}, {}) failed. expected={}, actual={}", input1, input2, expected, actual);
    }
}


int main() {
    SPDLOG_WARN("Running isSameTree tests:");
    TIMER_START(isSameTree);
    isSameTree_scaffold("[]", "[]");
    isSameTree_scaffold("[3,4,5,1,2]", "[4,1,2]");
    isSameTree_scaffold("[4,1,2]", "[4,1,2]");
    isSameTree_scaffold("[3,4,5,1,2,null,null,null,0]", "[4,1,2]");
    isSameTree_scaffold("[1,2]", "[1,null,2]");
    isSameTree_scaffold("[1,2,1]", "[1,1,2]");
    TIMER_STOP(isSameTree);
    SPDLOG_WARN("isSameTree tests using {} ms", TIMER_MSEC(isSameTree));

    SPDLOG_WARN("Running isSubtree tests:");
    TIMER_START(isSubtree);
    isSubtree_scaffold("[]", "[]", true);
    isSubtree_scaffold("[4,1,2]", "[4,1,2]", true);
    isSubtree_scaffold("[3,4,5,1,2]", "[4,1,2]", true);
    isSubtree_scaffold("[3,4,5,1,2,null,null,null,0]", "[4,1,2]", false);
    isSubtree_scaffold("[1,2]", "[1,null,2]", false);
    isSubtree_scaffold("[1,2,1]", "[1,1,2]", false);
    TIMER_STOP(isSubtree);
    SPDLOG_WARN("isSubtree tests using {} ms", TIMER_MSEC(isSubtree));

    SPDLOG_WARN("Running isSymmetric tests:");
    TIMER_START(isSymmetric);
    isSymmetric_scaffold("[1,2,2,3,4,4,3]", true);
    isSymmetric_scaffold("[1,2,2,null,3,null,3]", false);
    isSymmetric_scaffold("[1,2,2,null,3,4]", false);
    TIMER_STOP(isSymmetric);
    SPDLOG_WARN("isSymmetric tests using {} ms", TIMER_MSEC(isSymmetric));

    return 0;
}
