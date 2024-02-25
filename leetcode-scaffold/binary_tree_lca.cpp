#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 235, 236, 865 */ 

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* subtreeWithAllDeepest(TreeNode* root);
    int getDistance(TreeNode* root, TreeNode* p, TreeNode* q);
};

int Solution::getDistance(TreeNode* root, TreeNode* p, TreeNode* q) {
/*
    Extension from lowestCommonAncestor question:
    given a binary tree, and two nodes p and q in the tree, find the distance between p and q.
    the distance is defined as the edges in the path that connects p and q.
    for example, given a tree root=[0,1,null,2,3], p=2, q=3, the path connected p and q is [2,1,3] so the distance is 2
*/
    // 1. first find the lca of p and q
    std::function<TreeNode*(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr || node == p || node == q) {
            return node;
        } else {
            auto l = dfs(node->left);
            auto r = dfs(node->right);
            if (l != nullptr && r != nullptr) {
                return node;
            } else {
                return l != nullptr ? l : r;
            }
        }
    };

    TreeNode* lca = dfs(root);

    // 2. use bfs to find the path between p and q
    int steps = 0;
    map<TreeNode*, int> mp; // node, steps from lca to node
    queue<TreeNode*> que; que.push(lca);
    while (mp.size() != 2) {
        for (int k=que.size(); k!=0; --k) {
            auto t = que.front(); que.pop();
            if (t == p || t == q) {
                mp[t] = steps;
            }
            if (t->left != nullptr) {
                que.push(t->left);
            }
            if (t->right != nullptr) {
                que.push(t->right);
            }
        }
        steps++;
    }
    return mp[p] + mp[q];
}

TreeNode* Solution::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
/*
    Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
    According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between 
    two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow 
    a node to be a descendant of itself).”
    Note:
        All of the nodes' values will be unique.
        p and q are different and both values will exist in the binary tree.
*/
    if (root == nullptr || root == p || root == q) { // trivial case
        return root;
    }
    auto l = lowestCommonAncestor(root->left, p, q);
    auto r = lowestCommonAncestor(root->right, p, q);
    if (l != nullptr && r != nullptr) {
        return root;
    } else {
        return l != nullptr ? l : r;
    }
}

TreeNode* Solution::BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
/*
    Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
    Constraints:
        All of the nodes' values will be unique.
        p and q are different and both values will exist in the BST.    
*/

{ // iterative solution
    TreeNode* x = root;
    int a = std::min(p->val, q->val);
    int b = std::max(p->val, q->val);
    while (x != nullptr) {
        if (x->val < a) {
            x = x->right;
        } else if (x->val > b) {
            x = x->left;
        } else {
            break;
        }
    }
    return x;
}

{ // recursive solution
    int a = std::min(p->val, q->val);
    int b = std::max(p->val, q->val);
    std::function<TreeNode*(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return node;
        } else if (node->val < a) {
            return dfs(node->right);
        } else if (node->val > b) {
            return dfs(node->left);
        } else {
            return node;
        }
    };
    return dfs(root);
}

}

TreeNode* Solution::subtreeWithAllDeepest(TreeNode* root) {
/*
    Given a binary tree rooted at root, the depth of each node is the shortest distance to the root.
    A node is deepest if it has the largest depth possible among any node in the entire tree.
    The subtree of a node is that node, plus the set of all descendants of that node.
    Return the node with the largest depth such that it contains all the deepest nodes in its subtree.
*/

{ // naive solution
    typedef std::pair<TreeNode*, int> element_type;
    // return the intermediate answer and the largest depth of leaves of subtree rooted at node
    std::function<element_type(TreeNode*, int)> dfs = [&] (TreeNode* node, int depth) {
        if (node == nullptr) {
            return make_pair(node, depth);
        }
        auto l = dfs(node->left, depth+1);
        auto r = dfs(node->right, depth+1);
        if (l.second == r.second) {
            return make_pair(node, l.second);
        } else {
            return (l.second > r.second) ? l : r;
        }
    };
    return dfs(root, 0).first;
}

{ // previous solution
    typedef std::pair<TreeNode*, int> element_type;
    std::function<element_type(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return std::make_pair(node, 0);
        }
        auto l = dfs(node->left);
        auto r = dfs(node->right);
        // 1. if both subtrees of current node got the same depth, take current node as intermediate
        // 2. otherwise, take the result with larger depth as intermediate
        // 3. update depth for current node
        if (l.second == r.second) {
            return make_pair(node, l.second+1);
        } else if (l.second > r.second) {
            l.second++;
            return l;
        } else {
            r.second++;
            return r;
        }
    };
    return dfs(root).first;
}

}

void lowestCommonAncestor_scaffold() {
    std::string input = "[1,2,3,4,5,6,7,8]";
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    util::Log(logINFO) << "input: " << input; 

    Solution ss;
    util::Log(logINFO) << "lowestCommonAncestor_tester: "; 
    auto lowestCommonAncestor_tester = [&](TreeNode* p, TreeNode* q, TreeNode* expected) {
        TreeNode* ans = ss.lowestCommonAncestor(root, p, q);
        if (ans == expected) {
            util::Log(logINFO) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) passed";
        } else {
            util::Log(logERROR) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) failed, actual: " << ans->val;
        }
    };
    lowestCommonAncestor_tester(root->left->left->left, root->left->left, root->left->left);
    lowestCommonAncestor_tester(root->left->left->left, root->left, root->left);
    lowestCommonAncestor_tester(root->left->left->left, root->left->right, root->left);
    lowestCommonAncestor_tester(root->left->left->left, root->right->left, root);

    util::Log(logINFO) << "getDistance_tester: "; 
    auto getDistance_tester = [&](TreeNode* p, TreeNode* q, int expected) {
        int ans = ss.getDistance(root, p, q);
        if (ans == expected) {
            util::Log(logINFO) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected << ">) passed";
        } else {
            util::Log(logERROR) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected << ">) failed, actual: " << ans;
        }
    };
    getDistance_tester(root->left->left->left, root->left->left, 1);
    getDistance_tester(root->left->left->left, root->left, 2);
    getDistance_tester(root->left->left->left, root->left->right, 3);
    getDistance_tester(root->left->left->left, root->right->left, 5);
}

void BST_lowestCommonAncestor_scaffold() {
    std::string input = "[4,2,5,1,3,null,6]";
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    Solution ss;
    auto tester = [&](TreeNode* p, TreeNode* q, TreeNode* expected) {
        TreeNode* ans = ss.BST_lowestCommonAncestor(root, p, q);
        if (ans == expected) {
            util::Log(logINFO) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) passed";
        } else {
            util::Log(logERROR) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) failed, actual: " << ans->val;
        }
    };

    util::Log(logINFO) << "input: " << input; 
    tester(root->left->left, root->left, root->left);
    tester(root->left->left, root->left->right, root->left);
    tester(root->left->left, root->right, root);
    tester(root->left->left, root->right->right, root);
}

void subtreeWithAllDeepest_scaffold(std::string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    Solution ss;
    TreeNode* ans = ss.subtreeWithAllDeepest(root);
    if (ans->val == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") faild, actual: " << ans->val;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running lowestCommonAncestor tests:";
    TIMER_START(lowestCommonAncestor);
    lowestCommonAncestor_scaffold();
    TIMER_STOP(lowestCommonAncestor);
    util::Log(logESSENTIAL) << "lowestCommonAncestor using " << TIMER_MSEC(lowestCommonAncestor) << " milliseconds";

    util::Log(logESSENTIAL) << "Running BST_lowestCommonAncestor tests:";
    TIMER_START(BST_lowestCommonAncestor);
    BST_lowestCommonAncestor_scaffold();
    TIMER_STOP(BST_lowestCommonAncestor);
    util::Log(logESSENTIAL) << "BST_lowestCommonAncestor using " << TIMER_MSEC(BST_lowestCommonAncestor) << " milliseconds";

    util::Log(logESSENTIAL) << "Running subtreeWithAllDeepest tests:";
    TIMER_START(subtreeWithAllDeepest);
    subtreeWithAllDeepest_scaffold("[3,5,1,6,2,0,8,null,null,7,4]", 2);
    subtreeWithAllDeepest_scaffold("[4,2,5,1,3,null,6]", 4);
    subtreeWithAllDeepest_scaffold("[1,2,3,4,5,6,7,8]", 8);
    TIMER_STOP(subtreeWithAllDeepest);
    util::Log(logESSENTIAL) << "subtreeWithAllDeepest using " << TIMER_MSEC(subtreeWithAllDeepest) << " milliseconds";
}