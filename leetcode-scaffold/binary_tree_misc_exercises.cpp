#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 508, 968, 337, 979*/

class Solution {
public:
    std::vector<int> findFrequentTreeSum(TreeNode* root);
    int minCameraCover(TreeNode* root);
    int rob(TreeNode* root);
    int distributeCoins(TreeNode* root);
};

std::vector<int> Solution::findFrequentTreeSum(TreeNode* root) {
/*
    Given the root of a tree, you are asked to find the most frequent subtree sum. 
    The subtree sum of a node is defined as the sum of all the node values formed by
    the subtree rooted at that node (including the node itself). 
    So what is the most frequent subtree sum value? 
    If there is a tie, return all the values with the highest frequency in any order.
    Examples 1
        Input:
         5
        /  \
        2   -3
        return [2, -3, 4], since all the values happen only once, return all of them in any order.
    Examples 2
        Input:
        5
        /  \
        2   -5
        return [2], since 2 happens twice, however -5 only occur once.
*/

    std::vector<int> ans;
    if (root == nullptr) {
        return ans;
    }
    int frequency = 0;
    std::map<int, int> mp; // value, frequency
    // return the sum of subtree rooted at node, traverse a tree in post-order
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        int s = node->val;
        if (node->left != nullptr) {
            s+= dfs(node->left);
        }
        if (node->right != nullptr) {
            s+= dfs(node->right);
        }
        mp[s]++;
        frequency = std::max(frequency, mp[s]);
        return s;
    };
    dfs(root);
    for (auto it: mp) {
        if (it.second == frequency) {
            ans.push_back(it.first);
        }
    }
    return ans;
}


int Solution::minCameraCover(TreeNode* root) {
/*
    Given a binary tree, we install cameras on the nodes of the tree.
    Each camera at a node can monitor its parent, itself, and its immediate children.
    Calculate the minimum number of cameras needed to monitor all nodes of the tree.
    Example1:
        Input: [0,0,null,0,0]
        Output: 1
        Explanation: One camera is enough to monitor all nodes if placed as shown.
    Example2:
        Input: [0,0,null,0,null,0,null,null,0]
        Output: 2
        Explanation: At least two cameras are needed to monitor all nodes of the tree.
*/
    enum State {
        None,  // undefined status
        Camera, // a camera will be installed in node
        Covered // either left or right child has a camera installed
    };
    int ans = 0;
    // return the State of node, traversing the subtree rooted in node in post-order
    std::function<State(TreeNode*)> dfs = [&] (TreeNode* node) {
        // null node will always be covered
        if (node == nullptr) {
            return State::Covered;
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        if (l==State::Camera || r==State::Camera) {
            return State::Covered;
        } else if (l==State::None || r==State::None) {
            ans++;
            return State::Camera;
        } else {
            return State::None;
        }
    };
    if (dfs(root) == State::None) {
        // for case [0]
        ans++;
    }
    return ans;
}


int Solution::rob(TreeNode* root) {
/*
    The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the “root.” Besides the root, each house has one and only one parent house. 
    After a tour, the smart thief realized that “all houses in this place forms a binary tree”. It will automatically contact the police if two directly-linked houses were broken into on the same night.
    Determine the maximum amount of money the thief can rob tonight without alerting the police.

    Example 1: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
         3
        / \
        2   3
        \   \ 
        3   1
*/
{
    // optimization: add memoization
    map<TreeNode*, int> mp;
    // post-order traversal usage
    // return the maximum amount of money the thief can rob in subtree rooted at node
    std::function<int(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) { // trivial case
            return 0;
        }
        if (mp.count(node)) {
            return mp[node];
        }
        int p1 = node->val;
        if (node->left != nullptr) {
            p1 += rob(node->left->left);
            p1 += rob(node->left->right);
        }
        if (node->right != nullptr) {
            p1 += rob(node->right->left);
            p1 += rob(node->right->right);
        }
        int p2 = rob(node->left) + rob(node->right);
        mp[node] = max(p1, // how much money the thief could rob if he robbed node
                p2 // how much money the thief could rob if he didn't rob node
            );
        return mp[node];
    };
    return dfs(root);
}


{ // naive solution
    if (root == nullptr) { // trivial case
        return 0;
    }
    int p1 = root->val;
    if (root->left != nullptr) {
        p1 += rob(root->left->left);
        p1 += rob(root->left->right);
    }
    if (root->right != nullptr) {
        p1 += rob(root->right->left);
        p1 += rob(root->right->right);
    }
    int p2 = rob(root->left) + rob(root->right);
    return std::max(p1, // money the thieft can get if he has robbed at root
                p2); // money the thieft can get if he has not
}

}


int Solution::distributeCoins(TreeNode* root) {
/*
    You are given the root of a binary tree with n nodes where each node in the tree has `node.val` coins and there are `n` coins total.
    In one move, we may choose two adjacent nodes and move one coin from one node to another. (A move may be from parent to child, or from child to parent.)
    Return the number of moves required to make every node have exactly one coin.

    Constraints:
        The number of nodes in the tree is n.
        0 <= Node.val <= n
        The sum of Node.val is n

    Input: root = [3,0,0]
    Output: 2
    Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.

    Input: root = [0,3,0]
    Output: 3
    Explanation: From the left child of the root, we move two coins to the root [taking two moves]. Then, we move one coin from the root of the tree to the right child.

    Compute the balance of left/right subtree, ans += abs(balance(left)) + abs(balance(right))
*/
    int ans = 0;
    using element_t = std::pair<int, int>; // number of nodes in the subtree, number of coins in the subtree
    // traverse the subtree rooted at node in post-order
    std::function<element_t(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return element_t(0, 0);
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        ans += abs(l.first-l.second);
        ans += abs(r.first-r.second);
        return element_t(l.first+r.first+1, l.second+r.second+node->val);
    };
    dfs(root);
    return ans;
}


void findFrequentTreeSum_scaffold(std::string input, std::string expected) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    std::vector<int> actual = ss.findFrequentTreeSum(root);
    std::vector<int> expectedResult = stringTo1DArray<int>(expected);
    std::sort(actual.begin(), actual.end());
    std::sort(expectedResult.begin(), expectedResult.end());
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input, expected, numberVectorToString(actual));
    }
}


void minCameraCover_scaffold(std::string input, int expected) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    int actual = ss.minCameraCover(root);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input, expected, actual);
    }
}


void rob_scaffold(std::string input, int expected) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    int actual = ss.rob(root);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input, expected, actual);
    }
}


void distributeCoins_scaffold(std::string input, int expected) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    int actual = ss.distributeCoins(root);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input, expected, actual);
    }
}


int main() {
    SPDLOG_WARN("Running findFrequentTreeSum tests:");
    TIMER_START(findFrequentTreeSum);
    findFrequentTreeSum_scaffold("[5,2,-3]", "[4,2,-3]");
    findFrequentTreeSum_scaffold("[5,2,-5]", "[2]");
    TIMER_STOP(findFrequentTreeSum);
    SPDLOG_WARN("findFrequentTreeSum tests use {} ms", TIMER_MSEC(findFrequentTreeSum));

    SPDLOG_WARN("Running minCameraCover tests:");
    TIMER_START(minCameraCover);
    minCameraCover_scaffold("[0,0,null,0,0]", 1);
    minCameraCover_scaffold("[0,0,null,0,null,0,null,null,0]", 2);
    minCameraCover_scaffold("[0]", 1);
    minCameraCover_scaffold("[]", 0);
    TIMER_STOP(minCameraCover);
    SPDLOG_WARN("minCameraCover tests use {} ms", TIMER_MSEC(minCameraCover));

    SPDLOG_WARN("Running rob tests:");
    TIMER_START(rob);
    rob_scaffold("[0,0,null,0,0]", 0);
    rob_scaffold("[0,0,null,0,null,0,null,null,0]", 0);
    rob_scaffold("[3,2,3,null,null,1]", 5);
    rob_scaffold("[3,2,3,null,3, null,1]", 7);
    rob_scaffold("[3,4,5,1,3,null,1]", 9);
    TIMER_STOP(rob);
    SPDLOG_WARN("rob tests use {} ms", TIMER_MSEC(rob));

    SPDLOG_WARN("Running distributeCoins tests:");
    TIMER_START(distributeCoins);
    distributeCoins_scaffold("[3,0,0]", 2);
    distributeCoins_scaffold("[0,3,0]", 3);
    distributeCoins_scaffold("[1,0,2]", 2);
    distributeCoins_scaffold("[1,0,0,null,3]", 4);
    TIMER_STOP(distributeCoins);
    SPDLOG_WARN("distributeCoins tests use {} ms", TIMER_MSEC(distributeCoins));
    
}