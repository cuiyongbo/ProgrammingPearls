#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 129, 257, 112, 113, 437, 124, 543, 687*/

class Solution {
public:
    int sumNumbers(TreeNode* root);
    std::vector<std::string> binaryTreePaths(TreeNode* root);
    bool hasPathSum(TreeNode* root, int sum);
    std::vector<std::vector<int>> pathSum(TreeNode* root, int sum);
    int pathSum_437(TreeNode* root, int sum);
    int maxPathSum(TreeNode* root);
    int diameterOfBinaryTree(TreeNode* root);
    int longestUnivaluePath(TreeNode* root);
};


int Solution::sumNumbers(TreeNode* root) {
/*
    Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
    An example is the root-to-leaf path 1->2->3 which represents the number 123. Find the total sum of all root-to-leaf numbers.
    Note: A leaf is a node with no children.
*/
    // return root-to-leaf path sum for tree rooted at node, traverse tree in pre-order
    std::function <int(TreeNode*, int)> dfs = [&](TreeNode* node, int curSum) {
        if (node == nullptr) {
            return 0;
        }
        curSum = curSum * 10 + node->val;
        if (node->is_leaf()) {
            return curSum;
        } else {
            return dfs(node->left, curSum) // left root-to-leaf path
                    + dfs(node->right, curSum); // right root-to-leaf path
        }
    };
    return dfs(root, 0);
}


std::vector<std::string> Solution::binaryTreePaths(TreeNode* root) {
/*
    Given a binary tree, return all root-to-leaf paths.
    Note: A leaf is a node with no children.
    for example, given an input [1,2,3,null,5], return {["1->2->5", "1->3"]}.
*/
    std::vector<std::string> ans;
    auto make_path = [&] (const std::vector<TreeNode*>& buffer) {
        std::string path;
        for (auto n: buffer) {
            path.append(std::to_string(n->val));
            path.append("->");
        }
        // erase final "->"
        path.pop_back(); path.pop_back();
        ans.push_back(path);
    };
    std::vector<TreeNode*> buffer;
    // traverse the tree in pre-order
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        buffer.push_back(node);
        if (node->is_leaf()) {
            make_path(buffer);
        }
        dfs(node->left);
        dfs(node->right);
        buffer.pop_back();

    };
    dfs(root);
    return ans;
}


bool Solution::hasPathSum(TreeNode* root, int sum) {
/*
    Given a binary tree and a sum, determine if the tree has a root-to-leaf path 
    such that adding up all the values along the path equals the given sum.
    Hint: preorder traversal
*/
    // traverse the tree in pre-order
    if (root == nullptr) { // trivial path
        return false;
    }
    // test root
    if (root->is_leaf()) {
        return root->val == sum;
    }
    // test left, right subtree if root failed to meet the condition
    return hasPathSum(root->left, sum-root->val)
            || hasPathSum(root->right, sum-root->val);
}


std::vector<std::vector<int>> Solution::pathSum(TreeNode* root, int sum) {
/*
    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum. 
    Hint: dfs with backtrace
*/
    std::vector<int> path;
    std::vector<std::vector<int>> ans;
    std::function<void(TreeNode*, int)> dfs = [&] (TreeNode* node, int cur) {
        if (node == nullptr) {
            return;
        }
        path.push_back(node->val);
        // test whether the root meets the condition
        if (node->is_leaf()) {
            if (cur+node->val == sum) {
                ans.push_back(path);
            }
        }
        // test whether the subtrees meet the condition
        dfs(node->left, cur+node->val);
        dfs(node->right, cur+node->val);
        path.pop_back();
        return;
    };
    dfs(root, 0);
    return ans;
}


int Solution::pathSum_437(TreeNode* root, int sum) {
/*
    You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value.
    The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).
*/
    // return the number of paths starting from node and sum to the K
    // traverse the tree in pre-order
    std::function<int(TreeNode*, int)> dfs = [&] (TreeNode* node, int K) {
        if (node == nullptr) {
            return 0;
        }
        int count = 0;
        if (node->val == K) {
            count++;
        }
        count += dfs(node->left, K - node->val);
        count += dfs(node->right, K - node->val);
        return count;
    };
    int ans = dfs(root, sum);
    if (root != nullptr) {
        // paths that doesn't contain root
        ans += pathSum_437(root->left, sum);
        ans += pathSum_437(root->right, sum);
    }
    return ans;
}

int Solution::maxPathSum(TreeNode* root) {
/*
    Given a non-empty binary tree, find the maximum path sum.
    For this problem, a path is defined as any sequence of nodes from some
    starting node to any node in the tree along the parent-child connections.
    The path must contain at least one node and does not need to go through the root.
    for example, given an input [1,2,3], return 6

    dfs(node) return max path sum:
        1. node must be used
        2. contain at most one child (parent-child relation constraint).
*/
    int ans = INT32_MIN;
    // return the maxPathSum for tree rooted at node, using at most one child subtree
    std::function <int (TreeNode*)> dfs = [&] (TreeNode* node) { // postorder traversal
        if (node == nullptr) {
            return INT32_MIN;
        }
        int l = dfs(node->left);
        int r = dfs(node->right);
        int pl = l>0 ? l : 0;
        int pr = r>0 ? r : 0;
        // use both children to update ans
        ans = std::max(ans, node->val+pl+pr);
        return node->val + std::max(pl, pr); // use at most one child when return
    };
    dfs(root);
    return ans;
}


int Solution::diameterOfBinaryTree(TreeNode* root) {
/*
    Given a binary tree, you need to compute the length of the diameter of the tree.
    The diameter of a binary tree is the number of edges of the longest path between any two
    nodes in a tree. This path may or may not pass through the root.

    dfs(node) return the node count of a longest path:
        path must contain node
        only one child can be used
*/
    int ans = 0;
    // return the number of nodes in the longest unidirectional path starting with node
    std::function <int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node==nullptr) {
            return 0;
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        ans = max(ans, 1+l+r-1);
        return 1 + max(l, r);
    };
    dfs(root);
    return ans;
}


int Solution::longestUnivaluePath(TreeNode* root) {
/*
    Given a binary tree, find the length of the longest path where each node in the path has the same value.
    This path may or may not pass through the root.

    The length of a path between two nodes is represented by the number of edges between two nodes.

    dfs(node) returns the number of nodes along the longest univalue path:
        1. must contain node
        2. at most one child subtree can be used
*/
    int ans = 0;
    // return the number of nodes alone the longest uni-value path starting with node
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        int l = dfs(node->left);
        int r = dfs(node->right);
        int pl = 0;
        int pr = 0;
        if (node->left != nullptr && node->val == node->left->val) {
            pl = l;
        }
        if (node->right != nullptr && node->val == node->right->val) {
            pr = r;
        }
        ans = max(ans, 1+pl+pr-1);
        return 1 + max(pl, pr);
    };
    dfs(root);
    return ans;
}


void hasPathSum_scaffold(std::string input, int sum, bool expected) {
    TreeNode* t = stringToTreeNode(input);
    Solution ss;
    bool actual = ss.hasPathSum(t, sum);
    if (actual == expected) {
        SPDLOG_INFO("Case ({}, {}) passed", input, sum);
    } else {
        SPDLOG_ERROR("Case ({}, {}) failed, expected: {}, actual: {}", input, sum, expected, actual);
    }
}


void pathSum_scaffold(std::string input, int sum, std::string expectedResult) {
    TreeNode* t = stringToTreeNode(input);
    Solution ss;
    std::vector<std::vector<int>> actual = ss.pathSum(t, sum);
    auto expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        SPDLOG_INFO("Case ({}, {}, expectedResult={}) passed", input, sum, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, {}, expectedResult={}) failed, actual:", input, sum, expectedResult);
        for (const auto& a: actual) {
            print_vector(a);
        }
    }
}


void pathSum_437_scaffold(std::string input, int sum, int expectedResult) {
    TreeNode* t = stringToTreeNode(input);
    Solution ss;
    int actual = ss.pathSum_437(t, sum);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, {}, expectedResult={}) passed", input, sum, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, {}, expectedResult={}) failed, actual:", input, sum, expectedResult);
    }
}


void maxPathSum_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);
    Solution ss;
    int actual = ss.maxPathSum(root);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void diameterOfBinaryTree_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.diameterOfBinaryTree(root);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void longestUnivaluePath_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.longestUnivaluePath(root);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case ({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case ({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void sumNumbers_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    int actual = ss.sumNumbers(root);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual: {}", input, expectedResult, actual);
    }
}


void binaryTreePaths_scaffold(std::string input, std::vector<std::string> expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    std::vector<std::string> actual = ss.binaryTreePaths(root);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}) passed", input);
    } else {
        SPDLOG_ERROR("Case({}) failed", input);
        std::cout << "expected:";
        print_vector(expectedResult);
        std::cout << "acutal:";
        print_vector(actual);
    }
}


int main() {

    SPDLOG_WARN("Running hasPathSum tests:");
    TIMER_START(hasPathSum);
    hasPathSum_scaffold("[]", 0, false);
    hasPathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, true);
    hasPathSum_scaffold("[1,2,3,4,5,6,7]", 10, true);
    TIMER_STOP(hasPathSum);
    SPDLOG_WARN("hasPathSum tests use {} ms", TIMER_MSEC(hasPathSum));

    SPDLOG_WARN("Running pathSum tests:");
    TIMER_START(pathSum);
    pathSum_scaffold("[1,1,1]", 2, "[[1,1],[1,1]]");
    pathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, "[[5,4,11,2]]");
    pathSum_scaffold("[]", 22, "[]");
    TIMER_STOP(pathSum);
    SPDLOG_WARN("pathSum tests use {} ms", TIMER_MSEC(pathSum));

    SPDLOG_WARN("Running pathSum_437 tests:");
    TIMER_START(pathSum_437);
    pathSum_437_scaffold("[]", 0,  0);
    pathSum_437_scaffold("[1,1,1]", 2, 2);
    pathSum_437_scaffold("[10,5,-3,3,2,null,11,3,-2,null,1]", 8, 3);
    pathSum_437_scaffold("[1,-2,-3,1,3,-2,null,-1]", -1, 4);
    TIMER_STOP(pathSum_437);
    SPDLOG_WARN("pathSum_437 tests use {} ms", TIMER_MSEC(pathSum_437));

    SPDLOG_WARN("Running maxPathSum tests:");
    TIMER_START(maxPathSum);
    maxPathSum_scaffold("[-3]", -3);
    maxPathSum_scaffold("[1,2,3]", 6);
    maxPathSum_scaffold("[1,2,3,4]", 10);
    maxPathSum_scaffold("[-10,9,20,null,null,15,7]", 42);
    maxPathSum_scaffold("[-10,9,20,null,null,15,7,8]", 50);
    TIMER_STOP(maxPathSum);
    SPDLOG_WARN("maxPathSum tests use {} ms", TIMER_MSEC(maxPathSum));

    SPDLOG_WARN("Running diameterOfBinaryTree tests:");
    TIMER_START(diameterOfBinaryTree);
    diameterOfBinaryTree_scaffold("[]", 0);
    diameterOfBinaryTree_scaffold("[1]", 0);
    diameterOfBinaryTree_scaffold("[1,2,3,4,5]", 3);
    diameterOfBinaryTree_scaffold("[-10,9,20,null,null,15,7]", 3);
    diameterOfBinaryTree_scaffold("[-10,9,20,null,null,15,7,8]", 4);
    TIMER_STOP(diameterOfBinaryTree);
    SPDLOG_WARN("diameterOfBinaryTree tests use {} ms", TIMER_MSEC(diameterOfBinaryTree));

    SPDLOG_WARN("Running longestUnivaluePath tests:");
    TIMER_START(longestUnivaluePath);
    longestUnivaluePath_scaffold("[1,2,3,4,5]", 0);
    longestUnivaluePath_scaffold("[5,5,5,5]", 3);
    longestUnivaluePath_scaffold("[1,4,5,4,4,5]", 2);
    longestUnivaluePath_scaffold("[5,4,5,1,1,5]", 2);
    TIMER_STOP(longestUnivaluePath);
    SPDLOG_WARN("longestUnivaluePath tests use {} ms", TIMER_MSEC(longestUnivaluePath));

    SPDLOG_WARN("Running sumNumbers tests:");
    TIMER_START(sumNumbers);
    sumNumbers_scaffold("[1,2,3]", 25);
    sumNumbers_scaffold("[1,2,3,4,6,null,8]", 388);
    TIMER_STOP(sumNumbers);
    SPDLOG_WARN("sumNumbers tests use {} ms", TIMER_MSEC(sumNumbers));

    SPDLOG_WARN("Running binaryTreePaths tests:");
    TIMER_START(binaryTreePaths);
    binaryTreePaths_scaffold("[1,2,3]", {"1->2","1->3"});
    binaryTreePaths_scaffold("[1,2,3,4,6,null,8]", {"1->2->4","1->2->6","1->3->8"});
    binaryTreePaths_scaffold("[1,2,3,null,5]", {"1->2->5","1->3"});
    TIMER_STOP(binaryTreePaths);
    SPDLOG_WARN("binaryTreePaths tests use {} ms", TIMER_MSEC(binaryTreePaths));

    return 0;
}
