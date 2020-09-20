#include "leetcode.h"

/*
leetcode: 101
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
For example, this binary tree [1,2,2,3,4,4,3] is symmetric.
*/

using namespace std;
using namespace osrm;

class Solution {
public:
    bool isSymmetric_recursive(TreeNode* root);
    bool isSymmetric_iterative(TreeNode* root);
};

bool Solution::isSymmetric_recursive(TreeNode* root) {
    function<bool(TreeNode*, TreeNode*)> isMirror = [&](TreeNode* t1, TreeNode* t2) {
        if (t1 == nullptr && t2 == nullptr) {
            return true;
        } else if (t1 == nullptr || t2 == nullptr) {
            return false;
        } else {
            return (t1->val == t2->val) 
                    && isMirror(t1->left, t2->right)
                    && isMirror(t1->right, t2->left);
        }
    };

    if (root == nullptr) {
        return true;
    } else {
        return isMirror(root->left, root->right);
    }
    //return isMirror(root, root);
}

bool Solution::isSymmetric_iterative(TreeNode* root) {
    if (root == nullptr) {
        return true;
    }
    queue<TreeNode*> q;
    q.push(root->left); q.push(root->right);
    while(!q.empty()) {
        auto t1 = q.front(); q.pop();
        auto t2 = q.front(); q.pop();
        if (t1==nullptr && t2==nullptr) {
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

void isSymmetric_scaffold(string input1, bool expected) {
    TreeNode* root = stringToTreeNode(input1);
    std::unique_ptr<TreeNode> guard(root);
    bool ans = false;
    Solution ss;
    ans = ss.isSymmetric_iterative(root);
    if (ans == expected) {
        util::Log(logINFO) << "iterative Case(" << input1 << ", " << expected << ") passed.";
    } else {
        util::Log(logERROR) << "iterative Case(" << input1 << ", " << expected << ") failed.";
    }
    ans = ss.isSymmetric_recursive(root);
    if (ans == expected) {
        util::Log(logINFO) << "recursive Case(" << input1 << ", " << expected << ") passed.";
    } else {
        util::Log(logERROR) << "recursive Case(" << input1 << ", " << expected << ") failed.";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running isSymmetric tests:";
    TIMER_START(isSymmetric);
    isSymmetric_scaffold("[1,2,2,3,4,4,3]", true);
    isSymmetric_scaffold("[1,2,2,null,3,null,3]", false);
    TIMER_STOP(isSymmetric);
    util::Log(logESSENTIAL) << "isSymmetric tests using " << TIMER_MSEC(isSymmetric) << "ms.";
}
