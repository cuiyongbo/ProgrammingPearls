#include "leetcode.h"

using namespace std;
using namespace osrm;

/*
leetcode 872
Consider all the leaves of a binary tree, from left to right order, the values of those leaves form a leaf value sequence.
For example, for a tree with node [3,5,1,6,2,9,8,null,null,7,4], the leaf value sequence is (6, 7, 4, 9, 8).

Two binary trees are considered leaf-similar if their leaf value sequence is the same.
Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.

Hint: perform a preorder/inorder/postorder traversal to get the leaf value sequence
*/

class Solution {
public:
    bool leafSimilar(TreeNode* root1, TreeNode* root2);
private:
    void getLeafValueSequence(TreeNode* root, vector<int>& seq);
};

bool Solution::leafSimilar(TreeNode* root1, TreeNode* root2) {
    vector<int> seq1, seq2;
    getLeafValueSequence(root1, seq1);
    getLeafValueSequence(root2, seq2);
    return seq1 == seq2;
}

void Solution::getLeafValueSequence(TreeNode* root, vector<int>& seq) {
    stack<TreeNode*> s;
    if(root != nullptr) {
        s.push(root);
    }
    // preorder traversal
    while(!s.empty()) {
        auto t = s.top(); s.pop();
        if(t->left == nullptr && t->right == nullptr) {
            seq.push_back(t->val);
        }

        if(t->right != nullptr) {
            s.push(t->right);
        }
        if(t->left != nullptr) {
            s.push(t->left);
        }
    }
}

void leafSimilar_scaffold(string input1, string input2, bool expectedResult) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);

    Solution ss;
    bool actual = ss.leafSimilar(t1, t2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed.";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running leafSimilar tests:";
    TIMER_START(leafSimilar);
    
    leafSimilar_scaffold("[1]", "[1]", true);
    leafSimilar_scaffold("[1]", "[2]", false);
    leafSimilar_scaffold("[1,2]", "[2,2]", true);
    leafSimilar_scaffold("[1,2,3]", "[1,3,2]", false);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,7,4]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", true);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,4,7]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", false);
    TIMER_STOP(leafSimilar);
    util::Log(logESSENTIAL) << "Running leafSimilar tests uses " << TIMER_MSEC(leafSimilar) << "ms.";
    return 0;
}
