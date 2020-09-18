#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 100, 572 */
class Solution {
public:
    bool isSubtree(TreeNode* s, TreeNode* t); 
    bool isSameTree(TreeNode* node, TreeNode* t);    
};

bool Solution::isSameTree(TreeNode* node, TreeNode* t) {
/*
    Given two binary trees, write a function to check if they are the same or not.
    Two binary trees are considered the same if they are structurally identical 
    and the nodes have the same value.
*/

    if(node == nullptr && t == nullptr) {
        return true;
    } else if(node == nullptr || t == nullptr) {
        return false;
    } else if(node->val != t->val) {
        return false;
    } else {
        return isSameTree(node->left, t->left) && isSameTree(node->right, t->right);
    }
}

bool Solution::isSubtree(TreeNode* s, TreeNode* t)  {
/*
    Given two non-empty binary trees s and t, check whether tree t 
    has exactly the same structure and node values with a subtree of s. 
    A subtree of s is a tree consists of a node in s and all of this node's descendants. 
    The tree s could also be considered as a subtree of itself.
*/

    if (s == nullptr && t == nullptr) {
        return true;
    }
    else if (s == nullptr || t == nullptr) {
        return false;
    }
 
    queue<TreeNode*> q; q.push(s);
    while (!q.empty()) {
        int size = q.size();
        for(int i=0; i<size; ++i) {
            auto sc = q.front(); q.pop();
            if (isSameTree(sc, t)) {
                return true;
            }
            if(sc->left != nullptr) {
                q.push(sc->left);
            }
            if(sc->right != nullptr) {
                q.push(sc->right);
            }
        }
    }
    return false;
}

void isSameTree_scaffold(string input1, string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);
    Solution ss;
    bool acutual = ss.isSameTree(t1, t2);
    bool expected = input1 == input2;
    if (expected == acutual) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
    }
}

void isSubtree_scaffold(string input1, string input2, bool expected) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);
    Solution ss;
    bool acutual = ss.isSubtree(t1, t2);
    if (expected == acutual) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running isSameTree tests:";
    TIMER_START(isSameTree);
    isSameTree_scaffold("[]", "[]");
    isSameTree_scaffold("[3,4,5,1,2]", "[4,1,2]");
    isSameTree_scaffold("[4,1,2]", "[4,1,2]");
    isSameTree_scaffold("[3,4,5,1,2,null,null,null,0]", "[4,1,2]");
    isSameTree_scaffold("[1,2]", "[1,null,2]");
    isSameTree_scaffold("[1,2,1]", "[1,1,2]");
    TIMER_STOP(isSameTree);
    util::Log(logESSENTIAL) << "isSameTree tests using " << TIMER_MSEC(isSameTree) << "ms.";

    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running isSubtree tests:";
    TIMER_START(isSubtree);
    isSubtree_scaffold("[]", "[]", true);
    isSubtree_scaffold("[4,1,2]", "[4,1,2]", true);
    isSubtree_scaffold("[3,4,5,1,2]", "[4,1,2]", true);
    isSubtree_scaffold("[3,4,5,1,2,null,null,null,0]", "[4,1,2]", false);
    isSubtree_scaffold("[1,2]", "[1,null,2]", false);
    isSubtree_scaffold("[1,2,1]", "[1,1,2]", false);
    TIMER_STOP(isSubtree);
    util::Log(logESSENTIAL) << "isSubtree tests using " << TIMER_MSEC(isSubtree) << "ms.";

    return 0;
}
