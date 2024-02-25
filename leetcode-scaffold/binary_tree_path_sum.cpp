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

{ // refined version, preorder traversal
    std::function <int(TreeNode*, int)> dfs = [&](TreeNode* node, int curSum) {
        if (node == nullptr) {
            return 0;
        }
        curSum = curSum * 10 + node->val;
        if (node->is_leaf()) {
            return curSum;
        } else {
            return dfs(node->left, curSum) + dfs(node->right, curSum);
        }
    };
    return dfs(root, 0);
}

}

std::vector<std::string> Solution::binaryTreePaths(TreeNode* root) {
/*
    Given a binary tree, return all root-to-leaf paths.
    Note: A leaf is a node with no children.
    for example, given an input [1,2,3,null,5], return {["1->2->5", "1->3"]}.
*/

    std::vector<std::string> ans;
    std::vector<TreeNode*> buffer;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        } else if (node->is_leaf()) {
            std::string ss;
            for (auto n: buffer) {
                ss += std::to_string(n->val);
                ss += "->";
            }
            ss += std::to_string(node->val);
            ans.push_back(ss);
        } else {
            buffer.push_back(node);
            dfs(node->left);
            dfs(node->right);
            buffer.pop_back();
        }
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

    if (root == nullptr) {
        return false;
    } else if (root->is_leaf()) {
        return root->val == sum;
    } else {
        return hasPathSum(root->left, sum - root->val) ||
                hasPathSum(root->right, sum - root->val);
    }
}

std::vector<std::vector<int>> Solution::pathSum(TreeNode* root, int sum) {
/*
    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum. 
    Hint: dfs with backtrace
*/

    std::vector<int> buffer;
    std::vector<std::vector<int>> ans;
    std::function<void(TreeNode*, int)> dfs = [&] (TreeNode* node, int cur) {
        if (node == nullptr) {
            return;
        } else if (node->is_leaf()) {
            if (node->val == cur) {
                buffer.push_back(cur);
                ans.push_back(buffer);
                buffer.pop_back();
            }
        } else {
            buffer.push_back(node->val);
            dfs(node->left, cur - node->val);
            dfs(node->right, cur - node->val);
            buffer.pop_back();
        }
    };
    dfs(root, sum);
    return ans;
}

int Solution::pathSum_437(TreeNode* root, int sum) {
/*
    You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value.
    The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).
*/

    // return the number of paths, whch must start from node and sum to the K
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

{ // naive version
    // return the max path sum, which must contain node, and use at most one child
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        int l = dfs(node->left);
        int r = dfs(node->right);
        return std::max(node->val, node->val + std::max(l, r));
    };
    if (root == nullptr) {
        return INT32_MIN;
    } else {
        int p = root->val + std::max(0, dfs(root->left)) + std::max(0, dfs(root->right));
        int l = maxPathSum(root->left);
        int r = maxPathSum(root->right);
        return std::max(p, std::max(l, r));
    }
}

{
    int ans = INT32_MIN;
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

{ // naive version
    // return node count of the longest path which contains node,
    // and use one child of node
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        return 1 + std::max(l, r);
    };
    if (root == nullptr) {
        return 0;
    } else {
        return std::max(1 + dfs(root->left) + dfs(root->right), 
                    std::max(diameterOfBinaryTree(root->left),
                             diameterOfBinaryTree(root->right))
                )-1;
    }
}

{ // refined version
    int ans = 0;
    // return path with max nodes, with at most one subtree used
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

    int ans = INT32_MIN;
    // return the number of node alone longestUnivaluePath which contains node and use at most one child
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        int pl=0;
        int pr = 0;
        if (node->left != nullptr && node->val == node->left->val) {
            pl = l;
        }
        if (node->right != nullptr && node->val == node->right->val) {
            pr = r;
        }
        ans = std::max(ans, pl+pr);
        return 1+std::max(pl, pr);
    };
    dfs(root);
    return ans;
}

void hasPathSum_scaffold(std::string input, int sum, bool expected) {
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

void pathSum_scaffold(std::string input, int sum, std::string expectedResult) {
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    std::vector<std::vector<int>> actual = ss.pathSum(t, sum);
    auto expected = stringTo2DArray<int>(expectedResult);
    if (actual == expected) {
        util::Log(logINFO) << "Case (" << input << ", " << sum << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ") failed";
    }
}

void pathSum_437_scaffold(std::string input, int sum, int expectedResult) {
    TreeNode* t = stringToTreeNode(input);

    Solution ss;
    int actual = ss.pathSum_437(t, sum);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case (" << input << ", " << sum << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << sum << ", " << expectedResult << ") failed";
    }
}

void maxPathSum_scaffold(std::string input, int expected) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.maxPathSum(root);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", " << expected << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed";
        util::Log(logERROR) << "expected: " << expected << ", actual: " << actual;
    }
}

void diameterOfBinaryTree_scaffold(std::string input, int expected) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.diameterOfBinaryTree(root);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", " << expected << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed, expected: " << expected << ", actual: " << actual;
    }
}

void longestUnivaluePath_scaffold(std::string input, int expected) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper(root);

    Solution ss;
    int actual = ss.longestUnivaluePath(root);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", " << expected << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed, expected: " << expected << ", actual: " << actual;
    }
}

void sumNumbers_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    int actual = ss.sumNumbers(root);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed, actual: " << actual;
    }
}

void binaryTreePaths_scaffold(std::string input, std::vector<std::string> expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    std::vector<std::string> actual = ss.binaryTreePaths(root);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ") failed";

        util::Log(logERROR) << "expected:";
        for (auto& s: expectedResult) {
            util::Log(logERROR) << s;
        }
        util::Log(logERROR) << "acutal:";
        for(auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running hasPathSum tests:";
    TIMER_START(hasPathSum);
    hasPathSum_scaffold("[]", 0, false);
    hasPathSum_scaffold("[5,4,8,11,null,13,4,7,2,null,null,null,1]", 22, true);
    hasPathSum_scaffold("[1,2,3,4,5,6,7]", 10, true);
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
    longestUnivaluePath_scaffold("[1,4,5,4,4,5]", 2);
    longestUnivaluePath_scaffold("[5,4,5,1,1,5]", 2);
    TIMER_STOP(longestUnivaluePath);
    util::Log(logESSENTIAL) << "longestUnivaluePath using " << TIMER_MSEC(longestUnivaluePath) << " milliseconds";

    util::Log(logESSENTIAL) << "Running sumNumbers tests:";
    TIMER_START(sumNumbers);
    sumNumbers_scaffold("[1,2,3]", 25);
    sumNumbers_scaffold("[1,2,3,4,6,null,8]", 388);
    TIMER_STOP(sumNumbers);
    util::Log(logESSENTIAL) << "Running sumNumbers tests uses" << TIMER_MSEC(sumNumbers) << "ms.";

    util::Log(logESSENTIAL) << "Running binaryTreePaths tests:";
    TIMER_START(binaryTreePaths);
    binaryTreePaths_scaffold("[1,2,3]", {"1->2","1->3"});
    binaryTreePaths_scaffold("[1,2,3,4,6,null,8]", {"1->2->4","1->2->6","1->3->8"});
    binaryTreePaths_scaffold("[1,2,3,null,5]", {"1->2->5","1->3"});
    TIMER_STOP(binaryTreePaths);
    util::Log(logESSENTIAL) << "Running binaryTreePaths tests uses" << TIMER_MSEC(binaryTreePaths) << "ms.";

    return 0;
}
